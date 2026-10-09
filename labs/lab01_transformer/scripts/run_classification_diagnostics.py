"""Create post-hoc classifier diagnostics from persisted test predictions.

This script never loads a model or evaluates a DataLoader. It only visualizes the
predictions already written by the final notebook run.
"""
from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve


LAB_ROOT = Path(__file__).resolve().parents[1]
INPUT = LAB_ROOT / "outputs/classification/early_stopping/metrics/predictions.csv"
OUTPUT = LAB_ROOT / "outputs/classification/early_stopping/diagnostics_v1"


def main():
    predictions = pd.read_csv(INPUT)
    required = {"text", "actual", "prediction", "spam_probability"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    precision, recall, thresholds = precision_recall_curve(
        predictions.actual, predictions.spam_probability)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision, color="#4C78A8", linewidth=2)
    axis.set(title="Крива Precision-Recall: збережені test-прогнози",
             xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(alpha=.3)
    figure.tight_layout()
    figure.savefig(figures / "precision_recall_curve.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, normalize, title in zip(
            axes, [None, "true"], ["Абсолютні значення", "Нормалізація за справжнім класом"]):
        ConfusionMatrixDisplay.from_predictions(
            predictions.actual, predictions.prediction, labels=[0, 1],
            display_labels=["ham", "spam"], normalize=normalize, ax=axis,
            colorbar=False, values_format="d" if normalize is None else ".2f")
        axis.set_title(title)
    figure.suptitle("Матриці помилок: збережені test-прогнози")
    figure.tight_layout()
    figure.savefig(figures / "confusion_matrices.png", dpi=150)
    plt.close(figure)

    errors = predictions[predictions.actual != predictions.prediction].copy()
    errors["error_type"] = errors.apply(
        lambda row: "false_positive" if row.actual == 0 else "false_negative", axis=1)
    errors["error_confidence"] = errors.apply(
        lambda row: row.spam_probability if row.prediction == 1 else 1 - row.spam_probability,
        axis=1)
    errors.sort_values(["error_type", "error_confidence"], ascending=[True, False]).to_csv(
        OUTPUT / "classification_errors.csv", index=False)

    summary = {
        "source": str(INPUT.relative_to(LAB_ROOT)),
        "rows": int(len(predictions)),
        "false_positives": int(((predictions.actual == 0) & (predictions.prediction == 1)).sum()),
        "false_negatives": int(((predictions.actual == 1) & (predictions.prediction == 0)).sum()),
        "note": "Post-hoc analysis only; no model or DataLoader was evaluated.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
