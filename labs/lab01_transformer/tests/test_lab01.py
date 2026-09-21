"""CPU-only structural checks. No training, sweeps or full test-set evaluation."""
from pathlib import Path
import math
import ast
import nbformat
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from labs.lab01_transformer.src.common.text import Vocabulary, PAD, UNK, BOS, EOS, tokenize
from labs.lab01_transformer.src.common.positional_encoding import PositionalEncoding
from labs.lab01_transformer.src.common.checkpoint import save_checkpoint, load_checkpoint
from labs.lab01_transformer.src.common.training import run_epoch, perplexity
from labs.lab01_transformer.src.classification.data import (
    inspect_data, clean_data, split_data, class_weights, create_loaders, choose_max_length,
)
from labs.lab01_transformer.src.classification.model import SpamClassifier
from labs.lab01_transformer.src.generation.model import CausalLanguageModel, causal_mask
from labs.lab01_transformer.src.generation.data import LanguageModelDataset
from labs.lab01_transformer.src.generation.generate import generate_ids
from labs.lab01_transformer.src.generation.evaluate import rouge_l, evaluate_continuations

LAB = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def test_tokenizer_training_vocabulary():
    vocabulary = Vocabulary.build(["Hello, hello!"], min_frequency=1)
    assert tokenize("Hello!") == ["hello", "!"]
    assert vocabulary.tokens[:4] == ["<pad>", "<unk>", "<bos>", "<eos>"]
    assert vocabulary.encode("unseen") == [UNK]
    assert vocabulary.encode("HELLO") == vocabulary.encode("hello")


def test_real_data_splits_and_padding():
    path = LAB / "data/raw/spam.csv"
    if not path.exists():
        pytest.skip("Local dataset is intentionally not version-controlled")
    raw, summary = inspect_data(path)
    splits = split_data(clean_data(raw))
    keys = [set(frame.key) for frame in splits.values()]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]
    repeat = split_data(clean_data(raw))
    for name in splits:
        assert splits[name].index.tolist() == repeat[name].index.tolist()
    vocabulary = Vocabulary.build(splits["train"].Message)
    max_len = choose_max_length(splits["train"].Message)
    loaders = create_loaders(splits, vocabulary, max_len, batch_size=2)
    ids, labels = next(iter(loaders["validation"]))
    assert ids.eq(PAD).dtype == torch.bool
    assert ids.eq(PAD).shape == ids.shape
    assert ids.shape[0] == labels.shape[0] == 2
    assert ids.shape[1] <= max_len
    assert summary["rows"] == len(raw)
    weights = class_weights(splits["train"].label)
    counts = torch.tensor(splits["train"].label.value_counts().sort_index().values)
    torch.testing.assert_close(weights * counts, torch.ones(2) * len(splits["train"]) / 2)


def test_normalized_duplicates_and_conflicts():
    frame = pd.DataFrame({"Message": ["Hello", "hello", "Bad", "BAD", ""],
                          "Category": ["ham", "ham", "ham", "spam", "ham"]})
    cleaned = clean_data(frame)
    assert cleaned.Message.tolist() == ["Hello"]


@pytest.mark.parametrize("d_model", [7, 8])
def test_positional_encoding(d_model):
    position = PositionalEncoding(d_model, 10)
    result = position(torch.zeros(2, 4, d_model))
    assert result.shape == (2, 4, d_model)
    torch.testing.assert_close(result[0, 0, 0::2], torch.zeros((d_model + 1) // 2))
    torch.testing.assert_close(result[0, 0, 1::2], torch.ones(d_model // 2))
    with pytest.raises(ValueError):
        position(torch.zeros(1, 11, d_model))


def test_classification_padding_and_pooling():
    model = SpamClassifier(20, 12, d_model=8, nhead=2, dropout=0).eval()
    ids = torch.tensor([[4, 5, 6]])
    padded = torch.tensor([[4, 5, 6, PAD, PAD]])
    with torch.no_grad():
        assert model(ids).shape == (1, 2)
        torch.testing.assert_close(model(ids), model(padded), atol=1e-6, rtol=1e-5)
    encoded = torch.tensor([[[1., 3.], [3., 5.], [999., 999.]]])
    torch.testing.assert_close(model.masked_mean(encoded, torch.tensor([[False, False, True]])),
                               torch.tensor([[2., 4.]]))
    with pytest.raises(ValueError):
        model(torch.zeros(1, 3, dtype=torch.long))


@pytest.mark.parametrize("model_class", [SpamClassifier, CausalLanguageModel])
def test_invalid_heads(model_class):
    with pytest.raises(ValueError):
        model_class(20, 12, d_model=9, nhead=2)


def test_shift_windows_and_padding():
    vocabulary = Vocabulary.build(["one two three four five six"], min_frequency=1)
    dataset = LanguageModelDataset(["one two three four five six"], vocabulary, max_len=3)
    expected = vocabulary.encode("one two three four five six") + [EOS]
    actual = []
    assert dataset[0][0][0] == BOS
    for inputs, targets in dataset:
        valid = targets.ne(PAD).sum().item()
        torch.testing.assert_close(inputs[1:valid], targets[:valid - 1])
        actual.extend(targets[targets.ne(PAD)].tolist())
    assert actual == expected  # No lost or duplicated target at a window boundary.


def test_causal_mask():
    mask = causal_mask(4)
    assert mask.dtype == torch.bool and mask.shape == (4, 4)
    for query in range(4):
        for key in range(4):
            assert mask[query, key].item() == (key > query)


@pytest.mark.parametrize("layers", [1, 2, 5])
def test_no_future_leakage(layers):
    torch.manual_seed(42)
    model = CausalLanguageModel(20, 12, d_model=8, nhead=2, num_layers=layers).eval()
    first = torch.tensor([[BOS, 4, 5, 6, 7, 8]])
    changed = torch.tensor([[BOS, 4, 5, 9, 10, 11]])
    with torch.no_grad():
        logits = model(first)
        assert logits.shape == (1, 6, 20)
        torch.testing.assert_close(logits[:, :3], model(changed)[:, :3], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(logits[:, :3], model(first[:, :3]), atol=1e-6, rtol=1e-5)
        padded = torch.tensor([[BOS, 4, 5, PAD, PAD]])
        torch.testing.assert_close(logits[:, :3], model(padded)[:, :3], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("model_class,task", [(SpamClassifier, "classification"),
                                           (CausalLanguageModel, "generation")])
def test_checkpoint_roundtrip(tmp_path, model_class, task):
    vocabulary = Vocabulary.build(["one two three"], min_frequency=1)
    model = model_class(len(vocabulary), 12, d_model=8, nhead=2).eval()
    inputs = torch.tensor([[BOS, 4, 5]])
    path = tmp_path / "untrained.pt"
    save_checkpoint(path, model, vocabulary, task, {"seed": 42}, {})
    restored, restored_vocabulary, payload = load_checkpoint(path, "cpu")
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), restored(inputs))
    assert restored_vocabulary.tokens == vocabulary.tokens
    assert payload["model_config"] == model.config


class UniformLanguageModel(torch.nn.Module):
    def forward(self, ids):
        return torch.zeros(*ids.shape, 10)


def test_token_weighted_loss_ignores_padding():
    ids = torch.tensor([[2, 4, 5], [2, 6, 0], [2, 7, 0]])
    targets = torch.tensor([[4, 5, 3], [6, 3, 0], [7, 3, 0]])
    result = run_epoch(UniformLanguageModel(), DataLoader(TensorDataset(ids, targets), batch_size=2),
                       "cpu", "generation")
    assert result["loss"] == pytest.approx(math.log(10))
    assert result["perplexity"] == pytest.approx(10)
    assert math.isinf(perplexity(1000))


def test_rouge_l():
    assert rouge_l(["a", "b"], ["a", "b"]) == 1
    assert rouge_l([], ["a"]) == 0
    assert rouge_l(["a", "c"], ["a", "b"]) == .5


class ScriptedLanguageModel(torch.nn.Module):
    def __init__(self, token):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.token = token
        self.config = {"max_len": 3}
        self.seen = []

    def forward(self, ids):
        self.seen.append(ids.tolist()[0])
        logits = torch.zeros(*ids.shape, 12)
        logits[..., self.token] = 10
        return logits


def test_autoregressive_rolling_context_and_eos():
    model = ScriptedLanguageModel(4)
    assert generate_ids(model, [BOS, 5, 6, 7], 3) == [4, 4, 4]
    assert model.seen == [[5, 6, 7], [6, 7, 4], [7, 4, 4]]
    assert generate_ids(ScriptedLanguageModel(EOS), [BOS], 3) == []


def test_real_reference_words_not_unk_ids():
    vocabulary = Vocabulary.build(["known known"], min_frequency=1)
    metrics, rows = evaluate_continuations(ScriptedLanguageModel(UNK), vocabulary,
        ["known unseen absent missing word"], context_length=1, continuation_length=4, sample_size=1)
    assert metrics["rouge_l"] == 0
    assert metrics["bleu"] == 0
    assert rows.reference.iloc[0] == "unseen absent missing word"


@pytest.mark.parametrize("name", ["01_spam_classification.ipynb", "02_text_generation.ipynb"])
def test_notebooks_valid_without_errors(name):
    notebook = nbformat.read(LAB / "notebooks" / name, as_version=4)
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, name, "exec")
            assert not any(output.output_type == "error" for output in cell.outputs)
            tree = ast.parse(cell.source)
            assert not any(isinstance(node, ast.Name) and node.id.startswith("RUN_")
                           for node in ast.walk(tree))
            if "manual-run" in cell.metadata.get("tags", []):
                assert notebook.cells[index - 1].cell_type == "markdown"
                assert "Ручний запуск" in notebook.cells[index - 1].source
                assert not any(isinstance(node, ast.If) for node in tree.body)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id in ("fit_classifier", "fit_language_model", "run_grid"):
                        arguments = {argument.arg: argument.value for argument in node.keywords}
                        assert "epochs" not in arguments
                        assert arguments["max_epochs"].id == "MAX_EPOCHS"
                        assert arguments["patience"].id == "PATIENCE"
