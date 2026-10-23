"""Full/replay resolution and notebook visualizations for Stable Diffusion."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .config import LAB_ROOT, GenerationSpec
from .generation import generate_image

STATUS_LIVE = "LIVE COMPUTATION"
STATUS_REPLAY = "RECOMPUTED FROM EVIDENCE"


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
                       pipeline=None, context=None, root: Path = LAB_ROOT) -> dict:
    """Generate in full mode or verify and resolve one committed replay artifact."""
    if mode == "full":
        if context is None:
            raise ValueError("Full mode requires pipeline context")
        record = generate_image(pipeline, spec, context, root)
        return {**record, "status": STATUS_LIVE, "resolved_path": str(root / record["image_path"])}
    if mode != "replay":
        raise ValueError("mode must be 'full' or 'replay'")
    row = _match_manifest(spec, manifest)
    path = root / row["public_image_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != row["image_sha256"]:
        raise ValueError(f"SHA-256 mismatch for {spec.case_id}")
    with Image.open(path) as image:
        image.verify()
    record = row.to_dict()
    record.update(status=STATUS_REPLAY, resolved_path=str(path), sha256_verified=True)
    return record


def resolve_group(specs: list[GenerationSpec], **kwargs) -> list[dict]:
    return [resolve_generation(spec, **kwargs) for spec in specs]


def records_table(records: list[dict]) -> pd.DataFrame:
    columns = ["case_id", "status", "seed", "num_inference_steps", "guidance_scale",
               "scheduler", "generation_time_seconds", "peak_cuda_memory_mb", "sha256_verified"]
    frame = pd.DataFrame(records)
    return frame.reindex(columns=[c for c in columns if c in frame])


def plot_generation_gallery(records: list[dict], title: str, columns: int = 3):
    rows = math.ceil(len(records) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, records):
        with Image.open(record["resolved_path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(f"{record['case_id']}\n{record['status']}", fontsize=9)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_runtime(records: list[dict]):
    frame = pd.DataFrame(records).sort_values("num_inference_steps")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frame.num_inference_steps, frame.generation_time_seconds, "o-")
    ax.set(title="Recorded runtime for controlled step experiment",
           xlabel="inference steps", ylabel="wall-clock seconds")
    ax.grid(alpha=.25)
    fig.tight_layout()
    return fig


def plot_performance_dashboard(records: list[dict]):
    frame = pd.DataFrame(records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = frame.experiment.astype("category").cat.codes
    axes[0].scatter(frame.num_inference_steps, frame.generation_time_seconds, c=colors, cmap="tab10")
    axes[0].set(title="Runtime observations", xlabel="steps", ylabel="seconds")
    axes[1].scatter(frame.generation_time_seconds, frame.peak_cuda_memory_mb, c=colors, cmap="tab10")
    axes[1].set(title="Runtime vs peak allocated CUDA memory", xlabel="seconds", ylabel="MiB")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    return fig
