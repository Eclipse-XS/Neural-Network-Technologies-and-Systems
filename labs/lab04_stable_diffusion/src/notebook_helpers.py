"""Full/replay resolution and notebook visualizations for Stable Diffusion."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .config import LAB_ROOT, GenerationSpec
from .generation import generate_image

STATUS_LIVE = "LIVE_GENERATED"
STATUS_CACHE = "CACHE_REUSED"
STATUS_REPLAY = "HISTORICAL_REPLAY"
STATUS_NOT_EXECUTED = "NOT_EXECUTED"
STATUS_FAILED = "FAILED"


def experiment_plan(specs: list[GenerationSpec]) -> pd.DataFrame:
    return pd.DataFrame([asdict(spec) for spec in specs])


def _match_manifest(spec: GenerationSpec, manifest: pd.DataFrame) -> pd.Series:
    candidates = manifest[manifest["case_id"] == spec.case_id]
    if len(candidates) != 1:
        raise ValueError(f"Expected one verified row for {spec.case_id}; found {len(candidates)}")
    row = candidates.iloc[0]
    checks = {"prompt": spec.prompt, "seed": spec.seed, "num_inference_steps": spec.num_inference_steps,
              "guidance_scale": spec.guidance_scale, "width": spec.width, "height": spec.height}
    for key, expected in checks.items():
        if row[key] != expected:
            raise ValueError(f"Evidence mismatch for {spec.case_id}: {key}")
    actual_negative = None if pd.isna(row["negative_prompt"]) else row["negative_prompt"]
    if actual_negative != spec.negative_prompt:
        raise ValueError(f"Evidence mismatch for {spec.case_id}: negative_prompt")
    return row


def resolve_generation(spec: GenerationSpec, *, mode: str, manifest: pd.DataFrame,
                       pipeline=None, context=None, root: Path = LAB_ROOT,
                       force_regenerate: bool = False, run_id: str | None = None) -> dict:
    """Generate in full mode or verify and resolve one committed replay artifact."""
    if mode == "full":
        if context is None:
            raise ValueError("Full mode requires pipeline context")
        record = generate_image(pipeline, spec, context, root, force_regenerate=force_regenerate, run_id=run_id)
        resolved_path = str(root / record["image_path"])
        return {
            **record,
            "status": record.get("execution_status", STATUS_LIVE),
            "execution_mode": "full",
            "resolved_path": resolved_path,
            "sha256_verified": True
        }
    if mode != "replay":
        raise ValueError("mode must be 'full' or 'replay'")
    row = _match_manifest(spec, manifest)
    path = root / row["public_image_path"] if "public_image_path" in row and pd.notna(row["public_image_path"]) else root / row["image_path"]
    if not path.is_file():
        path = root / row["image_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["image_sha256"]:
        raise ValueError(f"SHA-256 mismatch for {spec.case_id}")
    with Image.open(path) as image:
        image.verify()
    record = row.to_dict()
    record.update(
        status=STATUS_REPLAY,
        execution_status=STATUS_REPLAY,
        execution_mode="replay",
        generated_in_current_run=False,
        pipeline_loaded_now=False,
        cache_hit=False,
        run_id=f"replay_{record.get('artifact_id', spec.case_id)}",
        verified_at=datetime.now(timezone.utc).isoformat(),
        resolved_path=str(path),
        sha256_verified=True
    )
    return record


def resolve_group(specs: list[GenerationSpec], **kwargs) -> list[dict]:
    return [resolve_generation(spec, **kwargs) for spec in specs]


def records_table(records: list[dict]) -> pd.DataFrame:
    columns = ["case_id", "status", "execution_status", "seed", "num_inference_steps", "guidance_scale",
               "scheduler", "generation_time_seconds", "peak_cuda_memory_mb", "sha256_verified",
               "generated_in_current_run"]
    frame = pd.DataFrame(records)
    return frame.reindex(columns=[c for c in columns if c in frame])


def plot_generation_gallery(records: list[dict], title: str, columns: int = 3):
    rows = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        with Image.open(record["resolved_path"]) as image:
            ax.imshow(image.convert("RGB"))
        status_label = record.get("execution_status") or record.get("status")
        ax.set_title(f"{record['case_id']}\n[{status_label}]", fontsize=8.5)
    fig.suptitle(title, fontweight="bold", fontsize=11)
    fig.tight_layout()
    return fig


def plot_runtime(records: list[dict]):
    frame = pd.DataFrame(records).sort_values("num_inference_steps")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(frame.num_inference_steps, frame.generation_time_seconds, "o-", color="#1f77b4", linewidth=2, markersize=6)
    for _, row in frame.iterrows():
        ax.annotate(f"{row.generation_time_seconds:.2f}s",
                    (row.num_inference_steps, row.generation_time_seconds),
                    textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8.5)
    ax.set(title="Inference Steps vs Wall-Clock Runtime (s)",
           xlabel="Number of Inference Steps", ylabel="Generation Time (seconds)")
    ax.set_xticks([20, 30, 50])
    ax.grid(alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig


def plot_performance_dashboard(records: list[dict]):
    frame = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    
    # 1. Steps vs Runtime
    frame_steps = frame.sort_values("num_inference_steps")
    axes[0].scatter(frame_steps.num_inference_steps, frame_steps.generation_time_seconds,
                    c="#1f77b4", s=50, edgecolors="black", alpha=0.8)
    axes[0].set(title="Runtime vs Inference Steps", xlabel="Inference Steps", ylabel="Runtime (s)")
    axes[0].set_xticks([20, 30, 50])
    axes[0].grid(alpha=0.25, linestyle="--")
    
    # 2. Runtime distribution
    axes[1].hist(frame.generation_time_seconds, bins=8, color="#2ca02c", edgecolor="black", alpha=0.7)
    median_val = float(frame.generation_time_seconds.median())
    axes[1].axvline(median_val, color="red", linestyle="--", label=f"Median: {median_val:.2f}s")
    axes[1].set(title="Runtime Distribution Across Experiments", xlabel="Runtime (s)", ylabel="Count")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25, linestyle="--")
    
    # 3. Execution Status counts
    status_series = frame["execution_status"] if "execution_status" in frame else frame["status"]
    status_counts = status_series.value_counts()
    status_counts.plot(kind="bar", ax=axes[2], color="#4a7c59", edgecolor="black", alpha=0.8)
    axes[2].set(title="Provenance / Status Breakdown", xlabel="Execution Status", ylabel="Image Count")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].grid(alpha=0.25, linestyle="--")
    
    fig.tight_layout()
    return fig
