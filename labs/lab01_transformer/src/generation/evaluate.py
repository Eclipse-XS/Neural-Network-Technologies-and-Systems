import random
from pathlib import Path
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from ..common.text import BOS, tokenize
from ..common.training import run_epoch
from .generate import generate_ids
from .data import create_loaders


def evaluate_language_model(model, loader, device):
    return run_epoch(model, loader, device, "generation")


def rouge_l(prediction, reference):
    """Token-level LCS F1 without stemming; empty prediction scores zero."""
    previous = [0] * (len(reference) + 1)
    for token in prediction:
        current = [0]
        for j, reference_token in enumerate(reference, 1):
            current.append(previous[j - 1] + 1 if token == reference_token
                           else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(prediction) + len(reference)) if prediction or reference else 0.0


def evaluate_continuations(model, vocabulary, texts, context_length=8,
                           continuation_length=12, sample_size=64, seed=42):
    if min(context_length, continuation_length, sample_size) < 1:
        raise ValueError("Lengths and sample_size must be positive")
    # Keep references as real words, NOT vocabulary IDs: shared UNKs must not earn credit.
    eligible = [tokenize(text) for text in texts
                if len(tokenize(text)) >= context_length + continuation_length]
    if not eligible:
        raise ValueError("No held-out documents long enough for this evaluation protocol")
    selected = random.Random(seed).sample(eligible, min(sample_size, len(eligible)))
    predictions, references, rows = [], [], []
    for tokens in selected:
        context = tokens[:context_length]
        reference = tokens[context_length:context_length + continuation_length]
        ids = [BOS] + vocabulary.encode(" ".join(context))
        generated = generate_ids(model, ids, continuation_length)
        prediction = [vocabulary.tokens[index] for index in generated]
        predictions.append(prediction)
        references.append(reference)
        rows.append({"context": " ".join(context), "reference": " ".join(reference),
                     "generated": " ".join(prediction)})
    metrics = {
        "bleu": float(corpus_bleu([[ref] for ref in references], predictions,
                                  smoothing_function=SmoothingFunction().method1)),
        "rouge_l": sum(rouge_l(p, r) for p, r in zip(predictions, references)) / len(rows),
        "examples": len(rows), "eligible_documents": len(eligible),
        "context_length": context_length, "continuation_length": continuation_length,
        "decoding": "greedy", "seed": seed,
    }
    return metrics, pd.DataFrame(rows)


def evaluate_final_generation(model, vocabulary, test_data, device, output_dir,
                              batch_size=32, context_length=8, continuation_length=12,
                              sample_size=64, seed=42):
    """Evaluate the selected LM with the existing token and continuation protocols."""
    loader = create_loaders({"test": test_data}, vocabulary, model.config["max_len"],
                             batch_size=batch_size, seed=seed)["test"]
    token_metrics = evaluate_language_model(model, loader, device)
    text_metrics, samples = evaluate_continuations(model, vocabulary, test_data.text,
        context_length=context_length, continuation_length=continuation_length,
        sample_size=sample_size, seed=seed)
    metrics = {**token_metrics, **text_metrics}
    output_dir = Path(output_dir)
    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).to_json(output_dir / "metrics" / "test.json", indent=2)
    samples.to_csv(output_dir / "samples" / "test_continuations.csv", index=False)
    return metrics, samples
