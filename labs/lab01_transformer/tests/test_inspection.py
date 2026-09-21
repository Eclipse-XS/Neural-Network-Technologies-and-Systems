"""Structural tests for the notebook-facing operations; no experiment execution."""
from types import SimpleNamespace
import json
import pandas as pd
import pytest
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from labs.lab01_transformer.src.common.text import Vocabulary
from labs.lab01_transformer.src.common.inspection import (
    summarize_splits, summarize_checkpoint, read_training_results,
)
from labs.lab01_transformer.src.common.visualization import (
    plot_dataset_overview, plot_positional_encoding, plot_causal_mask, plot_confusion_matrix,
    plot_history, plot_comparison,
)
from labs.lab01_transformer.src.common.positional_encoding import PositionalEncoding
from labs.lab01_transformer.src.classification.model import SpamClassifier
from labs.lab01_transformer.src.classification.inspection import inspect_classifier_pipeline
from labs.lab01_transformer.src.generation.model import CausalLanguageModel, causal_mask
from labs.lab01_transformer.src.generation.inspection import (
    inspect_generation_pipeline, inspect_autoregressive_batch, verify_causal_invariance,
)
from labs.lab01_transformer.src.generation.data import LanguageModelDataset
from labs.lab01_transformer.src.classification import evaluate as classification_evaluation
from labs.lab01_transformer.src.generation import evaluate as generation_evaluation


@pytest.mark.parametrize("model_class,inspect,output_shape", [
    (SpamClassifier, inspect_classifier_pipeline, (1, 2)),
    (CausalLanguageModel, inspect_generation_pipeline, (1, 3, 20)),
])
def test_pipeline_inspection_preserves_model_and_exposes_real_shapes(model_class, inspect, output_shape):
    model = model_class(20, 8, d_model=8, nhead=2).train()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    report = inspect(model, torch.tensor([[2, 4, 5]]))
    assert report.loc["embedding", "shape"] == (1, 3, 8)
    assert report.loc["positional_encoding", "shape"] == (1, 3, 8)
    assert report.loc["logits", "shape"] == output_shape
    assert model.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_causal_inspection_checks_all_variants_and_restores_mode():
    model = CausalLanguageModel(20, 8, d_model=8, nhead=2).train()
    first = torch.tensor([[2, 4, 5, 6]])
    changed = torch.tensor([[2, 4, 7, 8]])
    report = verify_causal_invariance(model, first, changed, prefix_length=2)
    assert report.index.tolist() == ["changed_suffix", "deleted_suffix", "padded_suffix"]
    assert report.passed.all() and model.training


def test_causal_inspection_rejects_future_leakage():
    class LeakingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, ids):
            return ids.float().sum(1)[:, None, None].expand(-1, ids.size(1), 2)

    model = LeakingModel().train()
    with pytest.raises(AssertionError):
        verify_causal_invariance(model, torch.tensor([[2, 4, 5]]),
                                 torch.tensor([[2, 4, 9]]), prefix_length=2)
    assert model.training


def test_shift_preview_and_split_summary():
    vocabulary = Vocabulary.build(["one two three"], min_frequency=1)
    inputs, targets = LanguageModelDataset(["one two three"], vocabulary, max_len=6)[0]
    preview = inspect_autoregressive_batch(inputs[None], targets[None], vocabulary)
    assert preview.input.iloc[0] == "<bos>" and preview.target.iloc[0] == "one"
    frames = {name: pd.DataFrame({"Message": [text], "key": [text], "Category": ["ham"], "label": [0]})
              for name, text in zip(["train", "validation", "test"], ["one two", "four", "five"])}
    split_summary, encoding = summarize_splits(frames, vocabulary, 6)
    assert split_summary.rows.tolist() == [1, 1, 1]
    assert encoding.vocabulary_size == len(vocabulary)
    frames["test"] = frames["train"]
    with pytest.raises(ValueError, match="disjoint"):
        summarize_splits(frames, vocabulary, 6)


def test_checkpoint_summary_does_not_invent_legacy_epoch():
    report = summarize_checkpoint({"model_config": {"d_model": 8}, "best_validation_metric": {"f1": .5}})
    assert pd.isna(report.loc["best_epoch"].iloc[0])


def test_persisted_history_and_comparison_dataframe_interfaces(tmp_path):
    # Schema fixtures only: no measured laboratory results are created.
    metric_dir = tmp_path / "metrics"
    metric_dir.mkdir()
    history = pd.DataFrame({"epoch": [1], "train_loss": [1.], "validation_loss": [1.],
                            "validation_accuracy": [.5], "validation_f1": [.5],
                            "validation_perplexity": [2.718]})
    history.to_csv(metric_dir / "history.csv", index=False)
    (metric_dir / "summary.json").write_text(json.dumps({"best_epoch": 1, "epochs_trained": 1}))
    loaded, summary = read_training_results(tmp_path)
    pd.testing.assert_frame_equal(loaded, history)
    assert summary.loc["best_epoch"].iloc[0] == 1
    comparison = pd.DataFrame({"d_model": [64], "nhead": [2], "validation_f1": [.5]})
    figures = [plot_history(loaded, "classification"), plot_history(loaded, "generation"),
               plot_comparison(comparison, "validation_f1")]
    assert all(figure.axes[0].get_title() for figure in figures)
    assert all(not plt.fignum_exists(figure.number) for figure in figures)


def test_figures_save_once_and_do_not_leave_automatic_displays(tmp_path):
    frame = pd.DataFrame({"Message": ["one", "two three"], "Category": ["ham", "spam"]})
    predictions = pd.DataFrame({"actual": [0, 1], "prediction": [0, 0]})
    figures = [plot_dataset_overview(frame, tmp_path / "data.png"),
               plot_dataset_overview(frame, include_classes=False),
               plot_positional_encoding(PositionalEncoding(8, 8)),
               plot_causal_mask(causal_mask(4)), plot_confusion_matrix(predictions)]
    assert (tmp_path / "data.png").is_file()
    for figure in figures:
        assert figure.axes[0].get_title() and figure.axes[0].get_xlabel()
        assert not plt.fignum_exists(figure.number)


def test_final_classifier_orchestration_without_evaluation(monkeypatch, tmp_path):
    vocabulary = Vocabulary.build(["one two"], min_frequency=1)
    frame = pd.DataFrame({"Message": ["one", "two"], "label": [1, 0]}, index=[8, 3])
    weights = torch.tensor([.5, 2.])
    calls = []

    def fake_evaluate(model, loader, device, actual_weights):
        assert loader.dataset.labels == [1, 0] and loader.batch_size == 2
        torch.testing.assert_close(actual_weights, weights)
        calls.append("metrics")
        return {"f1": .5}

    monkeypatch.setattr(classification_evaluation, "evaluate_classifier", fake_evaluate)
    monkeypatch.setattr(classification_evaluation, "predict_messages",
        lambda *args, **kwargs: pd.DataFrame({"prediction": [1, 1], "confidence": [.6, .7]}))
    metrics, predictions = classification_evaluation.evaluate_final_classifier(
        SimpleNamespace(config={"max_len": 4}), vocabulary, frame, "cpu", weights, tmp_path, batch_size=2)
    assert calls == ["metrics"] and metrics == {"f1": .5}
    assert predictions.actual.tolist() == [1, 0]
    assert (tmp_path / "metrics/test.json").is_file()
    assert (tmp_path / "metrics/predictions.csv").is_file()


def test_final_generation_orchestration_preserves_protocol(monkeypatch, tmp_path):
    vocabulary = Vocabulary.build(["one two"], min_frequency=1)
    frame = pd.DataFrame({"text": ["one two"], "document_id": ["article-1"]})

    def fake_continuations(model, vocab, texts, **settings):
        assert settings == dict(context_length=8, continuation_length=12, sample_size=64, seed=42)
        assert list(texts) == ["one two"]
        return {"bleu": 0., "rouge_l": 0.}, pd.DataFrame({"context": ["one"], "generated": [""]})

    monkeypatch.setattr(generation_evaluation, "evaluate_language_model", lambda *args: {"perplexity": 2.})
    monkeypatch.setattr(generation_evaluation, "evaluate_continuations", fake_continuations)
    metrics, samples = generation_evaluation.evaluate_final_generation(
        SimpleNamespace(config={"max_len": 4}), vocabulary, frame, "cpu", tmp_path)
    assert set(metrics) == {"perplexity", "bleu", "rouge_l"} and len(samples) == 1
    assert (tmp_path / "metrics/test.json").is_file()
    assert (tmp_path / "samples/test_continuations.csv").is_file()
