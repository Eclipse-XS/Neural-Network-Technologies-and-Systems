"""One explicit batch loop shared by both tasks, with correct loss denominators."""
import math
import json
import time
from pathlib import Path
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .text import PAD
from .checkpoint import save_checkpoint


def perplexity(loss):
    return math.exp(loss) if loss < 709 else math.inf


def run_epoch(model, loader, device, task, weights=None, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, denominator = 0.0, 0.0
    actual, predicted = [], []
    if weights is not None:
        weights = weights.to(device)
    with torch.set_grad_enabled(training):
        for input_ids, targets in loader:
            input_ids, targets = input_ids.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            if task == "classification":
                loss_sum = F.cross_entropy(logits, targets, weight=weights, reduction="sum")
                normalizer = weights[targets].sum() if weights is not None else targets.numel()
                actual.extend(targets.detach().cpu().tolist())
                predicted.extend(logits.detach().argmax(-1).cpu().tolist())
            else:
                loss_sum = F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                           ignore_index=PAD, reduction="sum")
                normalizer = targets.ne(PAD).sum()
            if training:
                (loss_sum / normalizer).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss_sum.detach().item()
            denominator += float(normalizer)
    if not denominator:
        raise ValueError("Cannot evaluate an empty loader")
    result = {"loss": total_loss / denominator}
    if task == "classification":
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, average="binary", pos_label=1, zero_division=0)
        result.update(accuracy=accuracy_score(actual, predicted),
                      precision=float(precision), recall=float(recall), f1=float(f1))
    else:
        result["perplexity"] = perplexity(result["loss"])
    return result


def fit(model, loaders, vocabulary, device, output_dir, task, weights=None,
        max_epochs=30, patience=5, learning_rate=1e-3, seed=42, dataset_metadata=None):
    """Keep the best validation state; stop after consecutive non-improving epochs.

    Strict improvement resets patience; ties count as non-improvement. The returned
    checkpoint is the selected model; the in-memory model remains at the last epoch.
    """
    if max_epochs < 1 or patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    if task not in ("classification", "generation"):
        raise ValueError("task must be classification or generation")
    model.to(device)
    output_dir = Path(output_dir)
    metric_dir = output_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoints" / "best.pt"
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=.01)
    history, best = [], -math.inf if task == "classification" else math.inf
    best_epoch, epochs_without_improvement = 0, 0
    best_validation = None
    start = time.perf_counter()
    training_config = dict(max_epochs=max_epochs, patience=patience,
                           learning_rate=learning_rate, seed=seed,
                           weight_decay=.01, batch_size=loaders["train"].batch_size,
                           class_weights=weights.tolist() if weights is not None else None)
    for epoch in range(1, max_epochs + 1):
        train = run_epoch(model, loaders["train"], device, task, weights, optimizer)
        validation = run_epoch(model, loaders["validation"], device, task, weights)
        if not math.isfinite(train["loss"]) or not all(math.isfinite(v) for v in validation.values()):
            raise FloatingPointError(f"Non-finite loss or validation metric at epoch {epoch}")
        row = {"epoch": epoch, "train_loss": train["loss"],
               **{f"validation_{key}": value for key, value in validation.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(metric_dir / "history.csv", index=False)
        score = validation["f1"] if task == "classification" else validation["loss"]
        improved = score > best if task == "classification" else score < best
        if improved:
            best = score
            best_epoch = epoch
            best_validation = validation.copy()
            epochs_without_improvement = 0
            save_checkpoint(checkpoint, model, vocabulary, task, training_config,
                            best_validation, best_epoch=best_epoch, dataset_metadata=dataset_metadata)
        else:
            epochs_without_improvement += 1
        print(row)
        if epochs_without_improvement >= patience:
            break
    summary = {"best_epoch": best_epoch, "epochs_trained": len(history),
               "stopped_early": len(history) < max_epochs,
               "stop_reason": "patience" if epochs_without_improvement >= patience else "max_epochs",
               "best_validation_metric": best_validation,
               "training_config": training_config,
               "seconds": time.perf_counter() - start,
               "parameters": sum(p.numel() for p in model.parameters())}
    (metric_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Best epoch: {best_epoch}; epochs trained: {len(history)}; stop: {summary['stop_reason']}")
    return {"checkpoint": str(checkpoint), "history": history, **summary}
