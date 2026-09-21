from pathlib import Path
import torch
from .text import Vocabulary


def save_checkpoint(path, model, vocabulary, task, training_config, best_validation_metric,
                    best_epoch=None, dataset_metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "model_config": model.config,
                "vocab": vocabulary.tokens, "tokenizer": "lowercase_regex_v1",
                "task": task,
                "training_config": training_config,
                "best_validation_metric": best_validation_metric,
                "best_epoch": best_epoch}
    if task == "classification":
        payload["label_mapping"] = {"ham": 0, "spam": 1}
    if dataset_metadata is not None:
        payload["dataset_metadata"] = dataset_metadata
    torch.save(payload, path)


def load_checkpoint(path, device="cpu", expected_dataset_metadata=None):
    from ..classification.model import SpamClassifier
    from ..generation.model import CausalLanguageModel
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["tokenizer"] != "lowercase_regex_v1":
        raise ValueError("Unsupported tokenizer")
    if expected_dataset_metadata is not None:
        if payload.get("task") != "generation" or payload.get("dataset_metadata") != expected_dataset_metadata:
            raise ValueError("Incompatible dataset/preprocessing checkpoint metadata")
        if payload["model_config"]["max_len"] != expected_dataset_metadata["max_len"]:
            raise ValueError("Checkpoint context length does not match corpus metadata")
        import hashlib
        vocab_hash = hashlib.sha256("\n".join(payload["vocab"]).encode()).hexdigest()
        if vocab_hash != expected_dataset_metadata["vocabulary_sha256"]:
            raise ValueError("Checkpoint vocabulary does not match corpus metadata")
    models = {"classification": SpamClassifier, "generation": CausalLanguageModel}
    model = models[payload["task"]](**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, Vocabulary(payload["vocab"]), payload
