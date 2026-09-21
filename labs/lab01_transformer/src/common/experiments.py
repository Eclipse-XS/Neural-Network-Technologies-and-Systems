from pathlib import Path
import math
import pandas as pd
from .reproducibility import seed_everything
from .checkpoint import load_checkpoint
from .training import fit

GRID_CONFIGS = ((64, 2), (64, 4), (128, 2), (128, 4))


def run_grid(model_class, base_config, loader_factory, vocabulary, device,
             output_dir, task, weights=None, max_epochs=30, patience=5,
             learning_rate=1e-3, seed=42, dataset_metadata=None):
    """Fixed data/order and early-stopping policy; validation-only selection."""
    rows = []
    output_dir = Path(output_dir)
    for d_model, nhead in GRID_CONFIGS:
        seed_everything(seed)
        config = {**base_config, "d_model": d_model, "nhead": nhead}
        model = model_class(**config)
        run_dir = output_dir / f"d{d_model}_h{nhead}"
        result = fit(model, loader_factory(seed=seed), vocabulary, device, run_dir,
                     task, weights, max_epochs=max_epochs, patience=patience,
                     learning_rate=learning_rate, seed=seed,
                     **({"dataset_metadata": dataset_metadata} if dataset_metadata is not None else {}))
        rows.append({"d_model": d_model, "nhead": nhead,
                     **{f"validation_{k}": v for k, v in result["best_validation_metric"].items()},
                     "best_epoch": result["best_epoch"],
                     "epochs_trained": result["epochs_trained"],
                     "stopped_early": result["stopped_early"],
                     "parameters": result["parameters"], "seconds": result["seconds"],
                     "checkpoint": str(Path(result["checkpoint"]).relative_to(output_dir))})
        pd.DataFrame(rows).to_csv(output_dir / "comparison.csv", index=False)
        del model
    return pd.DataFrame(rows)


def select_checkpoint(baseline_checkpoint, experiment_dir, task, expected_dataset_metadata=None):
    """Compare actual validation results; never select against test metrics."""
    if task not in ("classification", "generation"):
        raise ValueError("task must be classification or generation")
    candidates = []
    baseline_checkpoint, experiment_dir = Path(baseline_checkpoint), Path(experiment_dir)
    if baseline_checkpoint.exists():
        _, _, payload = load_checkpoint(baseline_checkpoint)
        candidates.append((baseline_checkpoint, payload["best_validation_metric"]))
    comparison = experiment_dir / "comparison.csv"
    if comparison.exists():
        for row in pd.read_csv(comparison).to_dict("records"):
            candidates.append((experiment_dir / row["checkpoint"],
                               {key.removeprefix("validation_"): value for key, value in row.items()
                                if key.startswith("validation_")}))
    if not candidates:
        raise FileNotFoundError("No checkpoints: run the baseline or grid cell first")
    metric = "f1" if task == "classification" else "loss"
    for path, metrics in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"Comparison refers to a missing checkpoint: {path}")
        if expected_dataset_metadata is not None:
            load_checkpoint(path, expected_dataset_metadata=expected_dataset_metadata)
        if not math.isfinite(metrics[metric]):
            raise ValueError(f"Non-finite validation {metric}: {path}")
    return sorted(candidates, key=lambda item: -item[1]["f1"] if task == "classification"
                  else item[1]["loss"])[0][0]
