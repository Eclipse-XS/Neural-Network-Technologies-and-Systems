"""Control-flow tests with scripted metrics; no optimizer steps or real training."""
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
from labs.lab01_transformer.src.common import training, experiments
from labs.lab01_transformer.src.common.checkpoint import load_checkpoint, save_checkpoint
from labs.lab01_transformer.src.common.text import Vocabulary
from labs.lab01_transformer.src.classification.model import SpamClassifier
from labs.lab01_transformer.src.classification.data import create_loaders
from labs.lab01_transformer.src.generation.model import CausalLanguageModel


def scripted_fit(monkeypatch, tmp_path, task, scores, max_epochs=20, patience=2):
    vocabulary = Vocabulary.build(["one two three"], min_frequency=1)
    model_class = SpamClassifier if task == "classification" else CausalLanguageModel
    model = model_class(len(vocabulary), 4, d_model=8, nhead=2)
    train_loader, validation_loader = SimpleNamespace(batch_size=2), object()
    calls = []

    class ValidationOnlyLoaders(dict):
        def __getitem__(self, key):
            assert key != "test", "Test must never participate in fitting"
            return super().__getitem__(key)

    def fake_epoch(model, loader, device, observed_task, weights=None, optimizer=None):
        assert device == "cpu" and observed_task == task
        calls.append(loader)
        epoch = (len(calls) + 1) // 2
        if optimizer is not None:
            assert loader is train_loader
            # Encode epoch identity in a weight to distinguish best from last state.
            with torch.no_grad():
                next(model.parameters()).fill_(epoch)
            return {"loss": 1.0}
        assert loader is validation_loader
        score = scores[epoch - 1]
        if task == "classification":
            return {"loss": float(epoch), "accuracy": .5, "precision": .5,
                    "recall": .5, "f1": score}
        return {"loss": score, "perplexity": training.perplexity(score)}

    monkeypatch.setattr(training, "run_epoch", fake_epoch)
    result = training.fit(model, ValidationOnlyLoaders(train=train_loader, validation=validation_loader),
                          vocabulary, "cpu", tmp_path, task,
                          max_epochs=max_epochs, patience=patience)
    return result, model, calls


@pytest.mark.parametrize("task,scores", [
    ("classification", [.4, .5, .49, .6, .6, .59]),
    ("generation", [5., 4., 4.1, 3., 3., 3.1]),
])
def test_patience_reset_exhaustion_and_best_checkpoint(monkeypatch, tmp_path, task, scores):
    result, last_model, calls = scripted_fit(monkeypatch, tmp_path, task, scores)
    assert result["best_epoch"] == 4
    assert result["epochs_trained"] == 6
    assert result["stopped_early"] and result["stop_reason"] == "patience"
    assert len(calls) == 12
    restored, _, payload = load_checkpoint(result["checkpoint"])
    assert payload["best_epoch"] == 4
    assert payload["training_config"]["max_epochs"] == 20
    assert payload["training_config"]["patience"] == 2
    assert torch.all(next(restored.parameters()) == 4)
    assert torch.all(next(last_model.parameters()) == 6)
    metric = "f1" if task == "classification" else "loss"
    assert payload["best_validation_metric"][metric] == scores[3]
    history = pd.read_csv(tmp_path / "metrics/history.csv")
    assert history.epoch.tolist() == list(range(1, 7))
    required = {"epoch", "train_loss", "validation_loss"}
    required |= ({"validation_accuracy", "validation_precision", "validation_recall", "validation_f1"}
                 if task == "classification" else {"validation_perplexity"})
    assert required <= set(history.columns)
    summary = json.loads((tmp_path / "metrics/summary.json").read_text())
    assert summary["best_epoch"] == 4 and summary["epochs_trained"] == 6


@pytest.mark.parametrize("task,scores", [
    ("classification", [.1, .2, .3]), ("generation", [3., 2., 1.]),
])
def test_max_epochs_caps_training(monkeypatch, tmp_path, task, scores):
    result, _, calls = scripted_fit(monkeypatch, tmp_path, task, scores, max_epochs=3, patience=5)
    assert result["epochs_trained"] == result["best_epoch"] == 3
    assert not result["stopped_early"] and result["stop_reason"] == "max_epochs"
    assert len(calls) == 6


@pytest.mark.parametrize("task,score", [("classification", 0.), ("generation", 1.)])
def test_first_epoch_saved_and_ties_exhaust_patience(monkeypatch, tmp_path, task, score):
    result, _, _ = scripted_fit(monkeypatch, tmp_path, task, [score, score], patience=1)
    assert result["best_epoch"] == 1 and result["epochs_trained"] == 2


@pytest.mark.parametrize("task", ["classification", "generation"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_selection_metric_fails_before_checkpoint(monkeypatch, tmp_path, task, invalid):
    with pytest.raises(FloatingPointError, match="Non-finite"):
        scripted_fit(monkeypatch, tmp_path, task, [invalid])
    assert not (tmp_path / "checkpoints/best.pt").exists()


@pytest.mark.parametrize("max_epochs,patience", [(0, 5), (30, 0), (-1, 5), (30, -1)])
def test_invalid_training_policy(max_epochs, patience, tmp_path):
    with pytest.raises(ValueError, match="positive"):
        training.fit(None, {}, None, "cpu", tmp_path, "classification",
                     max_epochs=max_epochs, patience=patience)


@pytest.mark.parametrize("task,model_class", [("classification", SpamClassifier),
                                            ("generation", CausalLanguageModel)])
def test_grid_preserves_policy_and_records_actual_epochs(monkeypatch, tmp_path, task, model_class):
    calls = []
    vocabulary = Vocabulary.build(["one two three"], min_frequency=1)

    def fake_fit(model, loaders, vocab, device, output_dir, observed_task, weights, **settings):
        assert set(loaders) == {"train", "validation"}
        assert observed_task == task and device == "cpu"
        assert settings == dict(max_epochs=30, patience=5, learning_rate=.002, seed=42)
        calls.append(model.config)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"checkpoint": str(Path(output_dir) / "checkpoints/best.pt"),
                "best_validation_metric": {"loss": 1., **({"f1": .5} if task == "classification"
                                                        else {"perplexity": training.perplexity(1.)})},
                "best_epoch": 3, "epochs_trained": 8, "stopped_early": True,
                "seconds": 0., "parameters": sum(p.numel() for p in model.parameters())}

    monkeypatch.setattr(experiments, "fit", fake_fit)
    frame = pd.DataFrame({"Message": ["one two"], "label": [0]})
    make_loaders = partial(create_loaders, {"train": frame, "validation": frame}, vocabulary, 4,
                           batch_size=2)
    result = experiments.run_grid(model_class, dict(vocab_size=len(vocabulary), max_len=4),
        make_loaders, vocabulary, "cpu", tmp_path, task,
        max_epochs=30, patience=5, learning_rate=.002, seed=42)
    assert [(config["d_model"], config["nhead"]) for config in calls] == [(64, 2), (64, 4), (128, 2), (128, 4)]
    assert all(config["d_model"] % config["nhead"] == 0 for config in calls)
    assert result.best_epoch.tolist() == [3] * 4
    assert result.epochs_trained.tolist() == [8] * 4
    assert len(pd.read_csv(tmp_path / "comparison.csv")) == 4


@pytest.mark.parametrize("task,scores", [("classification", [.3, .8, .5]),
                                       ("generation", [3., 1., 2.])])
def test_validation_selection_and_legacy_checkpoint(tmp_path, task, scores):
    vocabulary = Vocabulary.build(["one two three"], min_frequency=1)
    model_class = SpamClassifier if task == "classification" else CausalLanguageModel
    model = model_class(len(vocabulary), 4, d_model=8, nhead=2)
    baseline = tmp_path / "baseline/best.pt"
    grid_dir = tmp_path / "experiments"
    metric = "f1" if task == "classification" else "loss"
    save_checkpoint(baseline, model, vocabulary, task, {"epochs": 5}, {metric: scores[0]})
    # Simulate the actual old schema: neither max_epochs nor best_epoch was stored.
    payload = torch.load(baseline, weights_only=True)
    payload.pop("best_epoch")
    torch.save(payload, baseline)
    restored, _, restored_payload = load_checkpoint(baseline)
    assert restored.config == model.config and "best_epoch" not in restored_payload
    assert experiments.select_checkpoint(baseline, grid_dir, task) == baseline
    rows = []
    for index, score in enumerate(scores[1:]):
        path = grid_dir / str(index) / "best.pt"
        save_checkpoint(path, model, vocabulary, task, {}, {metric: score}, best_epoch=2)
        rows.append({"checkpoint": str(path.relative_to(grid_dir)), f"validation_{metric}": score})
    pd.DataFrame(rows).to_csv(grid_dir / "comparison.csv", index=False)
    assert experiments.select_checkpoint(baseline, grid_dir, task) == grid_dir / "0/best.pt"


def test_selection_reports_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="baseline or grid"):
        experiments.select_checkpoint(tmp_path / "baseline.pt", tmp_path, "classification")
