"""Run validation-only diagnostics for the existing BBC checkpoint; no training or test use."""
import json
from pathlib import Path
import sys
import time
import matplotlib
matplotlib.use("Agg")
import pandas as pd
import torch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))

from src.common.checkpoint import load_checkpoint
from src.common.reproducibility import seed_everything
from src.common.visualization import plot_decoding_comparison, plot_top_predictions, plot_unk_rates
from src.generation.data import inspect_corpus, clean_corpus, split_corpus, create_loaders
from src.generation.diagnostics import (compare_decoding_strategies,
    teacher_forced_prediction_profile, unk_rate)

DATA_PATH = LAB / "data/raw/bbc-text.csv"
CHECKPOINT = LAB / "outputs/generation/bbc/early_stopping/experiments/d128_h2/checkpoints/best.pt"
OUTPUT = LAB / "outputs/generation/bbc/diagnostics_v1"
SEED = 42

STRATEGIES = {
    "greedy": {"strategy": "greedy"},
    "temperature_0.8": {"strategy": "sample", "temperature": .8},
    "top_k_25": {"strategy": "sample", "temperature": .85, "top_k": 25},
    "top_p_0.9": {"strategy": "sample", "temperature": .85, "top_p": .9},
    "repetition_penalty": {"strategy": "greedy", "repetition_penalty": 1.2},
    "no_repeat_bigram": {"strategy": "greedy", "no_repeat_ngram_size": 2},
    "forbid_unk": {"strategy": "greedy", "forbid_unk": True},
    "combined_sampling": {"strategy": "sample", "temperature": .85, "top_k": 25,
                          "top_p": .9, "repetition_penalty": 1.2,
                          "no_repeat_ngram_size": 2, "forbid_unk": True},
}

QUALITY_REVIEW_NOTES = [
    "Penalty removes the greedy loop, but both alternatives drift from China's economy; combined output is more fluent but still incoherent.",
    "Greedy collapses immediately. Penalty and sampling form longer clauses but lose the royal-wedding topic.",
    "Penalty removes repeated UNK tokens but changes television coverage into unrelated economics; sampling also loses the topic.",
    "Penalty begins with the expected verb but then drifts. Sampling has fewer repetitions yet no reliable tax-policy continuation.",
    "Penalty and sampling avoid the UNK loop, but neither preserves the WTO negotiation context.",
    "Penalty produces a locally readable sentence fragment but substitutes unrelated entities; sampling behaves similarly.",
    "Penalty reduces repetition but loses the tsunami-aid topic. Combined sampling emits EOS before the first token.",
    "Both alternatives remove the greedy loop but replace the airline offering with unrelated phrases.",
    "Penalty is syntactically smoother than greedy but ignores the Mac Mini topic; sampling is diverse and off-topic.",
    "Penalty removes the loop but retains one UNK and drifts from politics; sampling is fluent locally but switches to sport/music.",
]


def quality_review_table(examples, count=10, seed=SEED):
    """Return predetermined examples for human review; no best-case filtering."""
    selected = examples[
        examples.strategy.isin(["greedy", "repetition_penalty", "combined_sampling"])
        & examples.seed.eq(seed) & examples.sample_index.lt(count)
    ]
    table = selected.pivot(index=["sample_index", "context", "reference"],
                           columns="strategy", values="generated").reset_index()
    table = table.rename(columns={"reference": "ground_truth",
        "repetition_penalty": "repetition_penalty_1.2"})
    table["review"] = QUALITY_REVIEW_NOTES[:len(table)]
    return table


def main():
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(exist_ok=True)
    print("=" * 60)
    print("EXPERIMENT: BBC Decoder | validation decoding diagnostics")
    print("=" * 60)
    print(f"Device         : {device}")
    print(f"Checkpoint     : {CHECKPOINT.relative_to(LAB)}")
    print("Data policy    : validation only; test untouched")
    started = time.perf_counter()
    frame = clean_corpus(inspect_corpus(DATA_PATH))
    splits = split_corpus(frame, seed=SEED)
    model, vocabulary, payload = load_checkpoint(CHECKPOINT, device=device)
    model.eval()
    loaders = create_loaders(splits, vocabulary, max_len=model.config["max_len"],
                             batch_size=32, seed=SEED, stride=model.config["max_len"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    corpus_rates = {name: unk_rate(split.text, vocabulary) for name, split in splits.items()}
    pd.Series(corpus_rates, name="unk_rate").to_csv(OUTPUT / "corpus_unk_rates.csv")
    plot_unk_rates(corpus_rates, figure_dir / "corpus_unk_rates.png")

    profile, top_tokens = teacher_forced_prediction_profile(
        model, loaders["validation"], vocabulary, device)
    (OUTPUT / "teacher_forced_profile.json").write_text(
        json.dumps(profile, indent=2, allow_nan=False), encoding="utf-8")
    top_tokens.to_csv(OUTPUT / "top_predicted_tokens.csv", index=False)
    plot_top_predictions(top_tokens, figure_dir / "top_predicted_tokens.png")

    summary, runs, examples = compare_decoding_strategies(
        model, vocabulary, splits["validation"].text, STRATEGIES,
        context_length=32, continuation_length=32, sample_size=32,
        selection_seed=SEED, sampling_seeds=(42, 43, 44))
    summary.to_csv(OUTPUT / "decoding_summary.csv", index=False)
    runs.to_csv(OUTPUT / "decoding_runs.csv", index=False)
    examples.to_csv(OUTPUT / "decoding_examples.csv", index=False)
    quality_review_table(examples).to_csv(
        OUTPUT / "generation_quality_review_10.csv", index=False)
    plot_decoding_comparison(summary, figure_dir / "decoding_comparison.png")

    prompt_rows = []
    for length in (8, 16, 32):
        result, _, _ = compare_decoding_strategies(
            model, vocabulary, splits["validation"].text,
            {"greedy": STRATEGIES["greedy"], "combined_sampling": STRATEGIES["combined_sampling"]},
            context_length=length, continuation_length=32, sample_size=16,
            selection_seed=SEED, sampling_seeds=(42, 43, 44))
        result.insert(1, "context_length", length)
        prompt_rows.append(result)
    pd.concat(prompt_rows, ignore_index=True).to_csv(OUTPUT / "prompt_length_comparison.csv", index=False)

    elapsed = time.perf_counter() - started
    run_info = {"dataset": "BBC News", "split": "validation", "checkpoint": str(CHECKPOINT),
                "tokenizer": payload["tokenizer"], "vocabulary_size": len(vocabulary),
                "model_config": payload["model_config"], "contexts": 32,
                "continuation_length": 32, "sampling_seeds": [42, 43, 44],
                "seconds": elapsed, "device": str(device),
                "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 2**20
                                       if device.type == "cuda" else None)}
    (OUTPUT / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Teacher-forced predicted UNK: {profile['predicted_unk_rate']:.2%}")
    print(f"Time           : {elapsed:.1f} s")
    if device.type == "cuda":
        print(f"Peak GPU memory: {run_info['peak_gpu_memory_mb']:.1f} MB")
    print(f"Artifacts      : {OUTPUT.relative_to(LAB)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
