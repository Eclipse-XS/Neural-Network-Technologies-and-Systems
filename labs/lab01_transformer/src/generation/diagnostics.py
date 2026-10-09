"""Validation-only diagnostics for decoding behaviour of a trained language model."""
from collections import Counter
import math
import random
import pandas as pd
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from ..common.text import BOS, EOS, PAD, UNK, tokenize
from .evaluate import rouge_l
from .generate import generate_ids


def unk_rate(texts, vocabulary):
    encoded = [token for text in texts for token in vocabulary.encode(text)]
    return sum(token == UNK for token in encoded) / len(encoded) if encoded else 0.0


def _ngram_repetition(tokens, n=2):
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def _distinct(predictions, n):
    grams = [tuple(tokens[i:i + n]) for tokens in predictions
             for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def text_quality_metrics(predictions, references, unk_token="<unk>", requested_length=None):
    if len(predictions) != len(references) or not predictions:
        raise ValueError("Predictions and references must be nonempty and aligned")
    token_count = sum(map(len, predictions))
    return {
        "bleu_4": float(corpus_bleu([[ref] for ref in references], predictions,
            smoothing_function=SmoothingFunction().method1)),
        "rouge_l": sum(rouge_l(pred, ref) for pred, ref in zip(predictions, references)) / len(predictions),
        "unk_rate": sum(token == unk_token for row in predictions for token in row) / token_count if token_count else 0.0,
        "repetition_rate": sum(_ngram_repetition(row) for row in predictions) / len(predictions),
        "distinct_1": _distinct(predictions, 1),
        "distinct_2": _distinct(predictions, 2),
        "average_length": token_count / len(predictions),
        "early_eos_rate": (sum(len(row) < requested_length for row in predictions) / len(predictions)
                           if requested_length else 0.0),
    }


def fixed_continuations(texts, context_length=32, continuation_length=32, sample_size=32, seed=42):
    eligible = [tokenize(text) for text in texts
                if len(tokenize(text)) >= context_length + continuation_length]
    if not eligible:
        raise ValueError("No documents satisfy the continuation protocol")
    selected = random.Random(seed).sample(eligible, min(sample_size, len(eligible)))
    return [(tokens[:context_length], tokens[context_length:context_length + continuation_length])
            for tokens in selected]


def compare_decoding_strategies(model, vocabulary, texts, strategies, *, context_length=32,
                                continuation_length=32, sample_size=32, selection_seed=42,
                                sampling_seeds=(42, 43, 44)):
    """Compare strategies on identical validation contexts; test data is not accepted."""
    samples = fixed_continuations(texts, context_length, continuation_length,
                                  sample_size, selection_seed)
    metric_rows, example_rows = [], []
    for name, config in strategies.items():
        seeds = sampling_seeds if config.get("strategy", "greedy") == "sample" else (selection_seed,)
        for run_seed in seeds:
            predictions, references = [], []
            for sample_index, (context, reference) in enumerate(samples):
                ids = [BOS] + vocabulary.encode(" ".join(context))
                generated = generate_ids(model, ids, continuation_length,
                                         seed=run_seed + sample_index, **config)
                prediction = [vocabulary.tokens[token] for token in generated]
                predictions.append(prediction)
                references.append(reference)
                example_rows.append({"strategy": name, "seed": run_seed,
                    "sample_index": sample_index, "context": " ".join(context),
                    "reference": " ".join(reference), "generated": " ".join(prediction)})
            metric_rows.append({"strategy": name, "seed": run_seed,
                **text_quality_metrics(predictions, references,
                                       requested_length=continuation_length)})
    runs = pd.DataFrame(metric_rows)
    summary = runs.groupby("strategy", sort=False).agg(
        **{column: (column, "mean") for column in runs.columns if column not in ("strategy", "seed")},
        runs=("seed", "size")).reset_index()
    return summary, runs, pd.DataFrame(example_rows)


@torch.no_grad()
def teacher_forced_prediction_profile(model, loader, vocabulary, device, max_batches=None):
    """Profile argmax tokens and confidence under teacher forcing on validation data."""
    model.eval()
    counts, target_counts = Counter(), Counter()
    probability_sum = entropy_sum = positions = 0.0
    for batch_index, (input_ids, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        input_ids, targets = input_ids.to(device), targets.to(device)
        logits = model(input_ids)
        valid = targets.ne(PAD)
        selected = logits[valid]
        probabilities = torch.softmax(selected, dim=-1)
        predicted = selected.argmax(-1)
        counts.update(predicted.cpu().tolist())
        target_counts.update(targets[valid].cpu().tolist())
        probability_sum += probabilities.max(-1).values.sum().item()
        entropy_sum += (-(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)).sum().item()
        positions += predicted.numel()
    if not positions:
        raise ValueError("No non-PAD validation targets")
    top = pd.DataFrame([{"token_id": token_id, "token": vocabulary.tokens[token_id], "count": count,
                         "share": count / positions, "target_count": target_counts[token_id]}
                        for token_id, count in counts.most_common(20)])
    metrics = {"positions": int(positions), "predicted_unk_rate": counts[UNK] / positions,
               "average_max_probability": probability_sum / positions,
               "average_entropy": entropy_sum / positions,
               "effective_candidates": math.exp(entropy_sum / positions)}
    return metrics, top
