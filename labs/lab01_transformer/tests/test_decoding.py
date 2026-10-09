import math
from pathlib import Path
import sys
import pytest
import torch
from torch import nn

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from src.common.text import Vocabulary, BOS, UNK
from src.generation.generate import generate_ids
from src.generation.diagnostics import text_quality_metrics, unk_rate


class DistributionModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.register_buffer("next_logits", torch.tensor(logits, dtype=torch.float))
        self.config = {"max_len": 8}

    def forward(self, input_ids):
        return self.next_logits.to(input_ids.device).view(1, 1, -1).expand(
            input_ids.size(0), input_ids.size(1), -1)


def test_sampling_is_deterministic_and_seed_sensitive():
    model = DistributionModel([-10, -10, -10, -10, 2.0, 1.5, 1.0])
    first = generate_ids(model, [BOS], 12, strategy="sample", temperature=.9, top_k=3, seed=7)
    assert first == generate_ids(model, [BOS], 12, strategy="sample", temperature=.9, top_k=3, seed=7)
    assert first != generate_ids(model, [BOS], 12, strategy="sample", temperature=.9, top_k=3, seed=8)


def test_forbid_unk_and_no_repeat_bigram():
    unk_model = DistributionModel([-10, 5, -10, -10, 4, 3])
    assert generate_ids(unk_model, [BOS], 1) == [UNK]
    assert generate_ids(unk_model, [BOS], 1, forbid_unk=True) == [4]
    repeated = generate_ids(DistributionModel([-10, -10, -10, -10, 5, 4]),
                            [BOS], 5, no_repeat_ngram_size=2)
    assert repeated[:3] == [4, 4, 5]


def test_repetition_penalty_handles_positive_and_negative_logits():
    positive = DistributionModel([-10, -10, -10, -10, 2.0, 1.9])
    negative = DistributionModel([-10, -10, -10, -10, -1.0, -1.1])
    assert generate_ids(positive, [BOS, 4], 1, repetition_penalty=1.2) == [5]
    assert generate_ids(negative, [BOS, 4], 1, repetition_penalty=1.2) == [5]


def test_special_tokens_and_default_greedy_are_preserved():
    model = DistributionModel([9, 8, 7, 10, 5])
    assert generate_ids(model, [BOS], 1) == generate_ids(model, [BOS], 1, strategy="greedy")
    assert generate_ids(model, [BOS], 1) == []  # PAD/BOS are banned; EOS terminates.


def test_top_k_and_top_p_filter_sampling_candidates():
    model = DistributionModel([-10, -10, -10, -10, 4.0, 3.0, 2.0])
    assert generate_ids(model, [BOS], 8, strategy="sample", top_k=1, seed=1) == [4] * 8
    assert generate_ids(model, [BOS], 8, strategy="sample", top_p=.5, seed=1) == [4] * 8


@pytest.mark.parametrize("kwargs", [
    {"strategy": "unknown"}, {"strategy": "sample", "temperature": 0},
    {"strategy": "sample", "top_k": 0}, {"strategy": "sample", "top_p": 0},
    {"repetition_penalty": .9},
])
def test_invalid_decoding_configuration(kwargs):
    with pytest.raises(ValueError):
        generate_ids(DistributionModel([-10] * 6), [BOS], 1, **kwargs)


def test_text_quality_metrics_and_corpus_unk_rate():
    predictions = [["a", "a", "a"], ["<unk>", "b"]]
    references = [["a", "b", "c"], ["b", "c"]]
    metrics = text_quality_metrics(predictions, references, requested_length=3)
    assert metrics["unk_rate"] == pytest.approx(.2)
    assert metrics["repetition_rate"] > 0
    assert metrics["distinct_1"] == pytest.approx(3 / 5)
    assert metrics["early_eos_rate"] == .5
    assert all(math.isfinite(value) for value in metrics.values())
    vocabulary = Vocabulary(["<pad>", "<unk>", "<bos>", "<eos>", "known"])
    assert unk_rate(["known missing"], vocabulary) == .5


def test_bleu_and_rouge_use_aligned_token_references():
    predictions = [["one", "two", "three", "four"], ["alpha", "beta", "gamma", "delta"]]
    aligned = text_quality_metrics(predictions, predictions)
    swapped = text_quality_metrics(predictions, list(reversed(predictions)))
    assert aligned["bleu_4"] == pytest.approx(1.0)
    assert aligned["rouge_l"] == pytest.approx(1.0)
    assert swapped["bleu_4"] < aligned["bleu_4"]
    assert swapped["rouge_l"] < aligned["rouge_l"]
