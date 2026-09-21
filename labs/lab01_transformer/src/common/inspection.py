"""Tabular inspection of data preparation and persisted experiment results."""
import json
from pathlib import Path
import pandas as pd
from .text import tokenize


def summarize_dataset(raw_inspection, cleaned_data):
    """Compare the source audit with the actual cleaning outcome."""
    report = {
        "raw.rows": raw_inspection["rows"],
        "raw.duplicate_texts": raw_inspection["duplicate_texts"],
        **{f"raw.missing.{key}": value for key, value in raw_inspection["missing"].items()},
        **{f"raw.count.{key}": value for key, value in raw_inspection["class_counts"].items()},
        **{f"raw.percent.{key}": value for key, value in raw_inspection["class_percent"].items()},
        **{f"raw.tokens.{key}": value for key, value in raw_inspection["lengths"].items()},
        "clean.rows": len(cleaned_data),
        "removed.rows": raw_inspection["rows"] - len(cleaned_data),
        **{f"clean.count.{key}": value for key, value in cleaned_data.Category.value_counts().items()},
    }
    return pd.Series(report, name="Значення").to_frame()


def summarize_splits(splits, vocabulary, max_len):
    """Validate disjoint normalized texts and report train-only encoding decisions."""
    seen = set()
    rows = []
    for name, frame in splits.items():
        keys = set(frame.key)
        if not len(frame) or seen & keys or len(keys) != len(frame):
            raise ValueError("Splits must be nonempty and contain disjoint unique texts")
        seen.update(keys)
        counts = frame.Category.value_counts()
        rows.append({"split": name, "rows": len(frame), "ham": counts.get("ham", 0),
                     "spam": counts.get("spam", 0), "spam_percent": 100 * frame.label.mean()})
    lengths = splits["train"].Message.map(lambda text: len(tokenize(text)))
    encoding = pd.Series({"vocabulary_size": len(vocabulary), "max_len": max_len,
                          "train_length_p95": lengths.quantile(.95),
                          "train_fraction_above_max_len": lengths.gt(max_len).mean()},
                         name="Train preprocessing")
    return pd.DataFrame(rows).set_index("split"), encoding


def read_training_results(run_dir):
    """Read real persisted history and flatten the nested run summary for display."""
    metric_dir = Path(run_dir) / "metrics"
    history = pd.read_csv(metric_dir / "history.csv")
    summary = json.loads((metric_dir / "summary.json").read_text(encoding="utf-8"))
    return history, pd.json_normalize(summary).T.rename(columns={0: "Значення"})


def summarize_checkpoint(metadata):
    report = {**{f"model.{key}": value for key, value in metadata["model_config"].items()},
              "best_epoch": metadata.get("best_epoch"),
              **{f"validation.{key}": value for key, value in metadata["best_validation_metric"].items()}}
    return pd.Series(report, name="Обрана модель").to_frame()
