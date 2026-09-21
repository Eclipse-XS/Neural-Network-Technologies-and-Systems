"""Corpus migration invariants; no model training or held-out evaluation."""
from pathlib import Path
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
from labs.lab01_transformer.src.common.text import Vocabulary, UNK, PAD, EOS
from labs.lab01_transformer.src.common.checkpoint import save_checkpoint, load_checkpoint
from labs.lab01_transformer.src.common import training, experiments
from labs.lab01_transformer.src.generation.data import (
    inspect_corpus, clean_corpus, split_corpus, create_loaders, corpus_metadata, LanguageModelDataset,
)
from labs.lab01_transformer.src.generation.corpus_inspection import summarize_corpus_splits
from labs.lab01_transformer.src.generation.model import CausalLanguageModel
from labs.lab01_transformer.src.generation.train import fit_language_model


@pytest.mark.parametrize("contents", ["Message,Category\nhello,ham\n", "category,text\nsport,hello,extra\n"])
def test_bbc_schema_rejects_invalid_csv(tmp_path, contents):
    path = tmp_path / "corpus.csv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        inspect_corpus(path)


def test_exact_dedup_preserves_text_and_first_metadata(tmp_path):
    path = tmp_path / "corpus.csv"
    path.write_text('category,text\nsport,"Hello, world"\ntech,"Hello, world"\ntech,"hello, world"\ntech,\n', encoding="utf-8")
    raw = inspect_corpus(path)
    clean = clean_corpus(raw)
    assert clean.text.tolist() == ["Hello, world", "hello, world"]
    assert clean.category.tolist() == ["sport", "tech"]
    assert clean.document_id.is_unique and len(raw) == 4
    assert clean.equals(clean_corpus(raw))


def test_document_splits_train_vocabulary_and_window_provenance():
    raw = pd.DataFrame({"text": [f"article unique{i} common common" for i in range(40)],
                        "category": ["topic_a", "topic_b"] * 20})
    clean = clean_corpus(pd.concat([raw, raw.iloc[:2]]))
    splits = split_corpus(clean)
    assert all(frame.equals(split_corpus(clean)[name]) for name, frame in splits.items())
    vocab = Vocabulary.build(splits["train"].text, min_frequency=1)
    assert "topic_a" not in vocab.indices and "topic_b" not in vocab.indices
    held_out_word = splits["test"].text.iloc[0].split()[1]
    assert vocab.encode(held_out_word) == [UNK]
    loaders = create_loaders(splits, vocab, max_len=3, stride=3, batch_size=2)
    seen = set()
    for name, loader in loaders.items():
        ids = set(loader.dataset.document_ids)
        assert ids == set(splits[name].document_id) and not ids & seen
        seen.update(ids)
    with pytest.raises(ValueError, match="overlapping"):
        create_loaders({"train": splits["train"], "test": splits["train"]}, vocab)
    changed = {name: frame.assign(category="irrelevant") for name, frame in splits.items()}
    other = create_loaders(changed, vocab, max_len=3)
    for name in loaders:
        for (x, y), (other_x, other_y) in zip(loaders[name].dataset, other[name].dataset):
            assert torch.equal(x, other_x) and torch.equal(y, other_y)


def test_windows_cover_each_target_once_including_eos():
    text = "one two three four five six seven"
    vocab = Vocabulary.build([text], min_frequency=1)
    dataset = LanguageModelDataset([text], vocab, max_len=3)
    targets = [token for _, target in dataset for token in target.tolist() if token != PAD]
    assert targets == vocab.encode(text) + [EOS]
    assert len(set(dataset.document_ids)) == 1
    with pytest.raises(ValueError, match="stride"):
        LanguageModelDataset([text], vocab, max_len=3, stride=1)


def test_real_bbc_preparation_counts_without_training():
    path = Path(__file__).parents[1] / "data/raw/bbc-text.csv"
    raw = inspect_corpus(path)
    assert len(raw) == 2225 and raw.text.duplicated().sum() == 99
    splits = split_corpus(clean_corpus(raw))
    vocabulary = Vocabulary.build(splits["train"].text, max_size=12000, min_frequency=2)
    summary = summarize_corpus_splits(splits, vocabulary, 64)
    assert summary.documents.tolist() == [1488, 319, 319]
    assert summary.windows.tolist() == [10669, 2230, 2284]
    assert len(vocabulary) == 12000


def test_bbc_metadata_propagation_and_incompatible_checkpoint_rejection(monkeypatch, tmp_path):
    source = tmp_path / "bbc.csv"
    source.write_text("category,text\ntech,one two three", encoding="utf-8")
    vocabulary = Vocabulary.build(["one two three"], min_frequency=1)
    metadata = corpus_metadata(source, vocabulary, dict(max_size=12000, min_frequency=1), 4, 4)
    model = CausalLanguageModel(len(vocabulary), 4, d_model=8, nhead=2)
    # Exercise the save path with scripted metrics: no optimizer step/forward/evaluation.
    monkeypatch.setattr(training, "run_epoch", lambda *args, **kwargs: {"loss": 1., "perplexity": 2.718})
    result = fit_language_model(model, {"train": SimpleNamespace(batch_size=2), "validation": object()},
                                vocabulary, "cpu", tmp_path / "baseline", max_epochs=1,
                                dataset_metadata=metadata)
    _, _, payload = load_checkpoint(result["checkpoint"], expected_dataset_metadata=metadata)
    assert payload["dataset_metadata"] == metadata and "label_mapping" not in payload
    for key, value in [("dataset", "other"), ("stride", 2), ("source_sha256", "different")]:
        with pytest.raises(ValueError, match="Incompatible"):
            load_checkpoint(result["checkpoint"], expected_dataset_metadata={**metadata, key: value})
    legacy = tmp_path / "legacy.pt"
    save_checkpoint(legacy, model, vocabulary, "generation", {}, {"loss": .5})
    with pytest.raises(ValueError, match="Incompatible"):
        load_checkpoint(legacy, expected_dataset_metadata=metadata)
    with pytest.raises(ValueError, match="Incompatible"):
        experiments.select_checkpoint(legacy, tmp_path / "empty", "generation", metadata)
    # Check grid metadata forwarding without fitting any grid model.
    observed = []
    def fake_fit(*args, **kwargs):
        observed.append(kwargs["dataset_metadata"])
        output = Path(args[4])
        output.mkdir(parents=True)
        return dict(checkpoint=str(output / "best.pt"), best_validation_metric={"loss": 1.},
                    best_epoch=1, epochs_trained=1, stopped_early=False, parameters=1, seconds=0.)
    monkeypatch.setattr(experiments, "fit", fake_fit)
    experiments.run_grid(CausalLanguageModel, model.config, lambda seed: {}, vocabulary,
                         "cpu", tmp_path / "grid", "generation", dataset_metadata=metadata)
    assert observed == [metadata] * 4
