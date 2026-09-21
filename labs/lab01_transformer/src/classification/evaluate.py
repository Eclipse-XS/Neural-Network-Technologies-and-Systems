import pandas as pd
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from ..common.text import PAD, UNK
from ..common.training import run_epoch
from .data import create_loaders


def evaluate_classifier(model, loader, device, weights=None):
    return run_epoch(model, loader, device, "classification", weights)


@torch.no_grad()
def predict_messages(model, vocabulary, texts, batch_size=32):
    model.eval()
    device = next(model.parameters()).device
    rows = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        tokens = [torch.tensor(vocabulary.encode(text)[:model.config["max_len"]] or [UNK])
                  for text in batch]
        input_ids = pad_sequence(tokens, batch_first=True, padding_value=PAD).to(device)
        probabilities = model(input_ids).softmax(-1).cpu()
        for text, probability in zip(batch, probabilities):
            rows.append({"text": text[:180], "prediction": probability.argmax().item(),
                         "confidence": probability.max().item(), "spam_probability": probability[1].item()})
    return pd.DataFrame(rows)


def evaluate_final_classifier(model, vocabulary, test_data, device, weights, output_dir,
                              batch_size=32, seed=42):
    """Evaluate a previously selected model and persist its metrics and predictions."""
    loader = create_loaders({"test": test_data}, vocabulary, model.config["max_len"],
                             batch_size=batch_size, seed=seed)["test"]
    metrics = evaluate_classifier(model, loader, device, weights)
    predictions = predict_messages(model, vocabulary, test_data.Message, batch_size=batch_size)
    predictions["actual"] = test_data.label.to_numpy()
    metric_dir = Path(output_dir) / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).to_json(metric_dir / "test.json", indent=2)
    predictions.to_csv(metric_dir / "predictions.csv", index=False)
    return metrics, predictions
