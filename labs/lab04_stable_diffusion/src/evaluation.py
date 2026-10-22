"""Evidence-based evaluation helpers for persisted Stable Diffusion experiments."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image


def load_generation_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {"case_id", "experiment", "prompt", "seed", "num_inference_steps",
                "guidance_scale", "scheduler", "generation_time_seconds", "image_path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    return frame


def validate_images(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for record in frame.to_dict("records"):
        path = Path(root) / record["image_path"]
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
        rows.append({"case_id": record["case_id"], "exists": path.exists(),
                     "width": width, "height": height, "valid_png": path.suffix.lower() == ".png"})
    return pd.DataFrame(rows)


def quality_rubric(observations_path: Path) -> pd.DataFrame:
    """Return explicitly manual, qualitative evidence; no invented CLIP score."""
    observations = json.loads(Path(observations_path).read_text(encoding="utf-8"))
    mapping = {
        "seed": ("composition_variability", "descriptive"),
        "steps": ("detail_and_runtime_tradeoff", "descriptive"),
        "cfg": ("prompt_strength_and_artifacts", "descriptive"),
        "negative_prompt_seed42": ("negative_prompt_pair", "descriptive"),
        "negative_prompt_seed123": ("negative_prompt_pair", "descriptive"),
        "styles": ("style_adherence", "descriptive"),
    }
    return pd.DataFrame([{"experiment": key, "criterion": mapping[key][0],
                          "assessment_type": mapping[key][1], "observation": value}
                         for key, value in observations.items()])


def scheduler_experiment_plan() -> pd.DataFrame:
    return pd.DataFrame([
        {"scheduler": "PNDMScheduler", "status": "EXECUTED_BASELINE", "steps": 30,
         "reason": "Scheduler recorded in all 13 verified sidecars."},
        {"scheduler": "EulerDiscreteScheduler", "status": "NOT_EXECUTED", "steps": 30,
         "reason": "No verified image or timing exists for this scheduler."},
        {"scheduler": "DDIMScheduler", "status": "NOT_EXECUTED", "steps": 30,
         "reason": "No verified image or timing exists for this scheduler."},
    ])


def performance_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame.groupby("experiment", as_index=False)
            .agg(observations=("case_id", "count"),
                 mean_runtime_s=("generation_time_seconds", "mean"),
                 std_runtime_s=("generation_time_seconds", "std"),
                 min_runtime_s=("generation_time_seconds", "min"),
                 max_runtime_s=("generation_time_seconds", "max")))
