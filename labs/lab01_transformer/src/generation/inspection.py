"""Shift, decoder shapes and causal-invariance demonstrations on small inputs."""
import math
import pandas as pd
import torch
from ..common.text import PAD
from .model import causal_mask


def inspect_autoregressive_batch(input_ids, targets, vocabulary, limit=16):
    if input_ids.shape != targets.shape or input_ids.ndim != 2:
        raise ValueError("Expected matching [batch, sequence] inputs and targets")
    valid = targets[0].ne(PAD).sum().item()
    torch.testing.assert_close(input_ids[0, 1:valid], targets[0, :valid - 1])
    return pd.DataFrame({"input": [vocabulary.tokens[i] for i in input_ids[0, :limit].tolist()],
                         "target": [vocabulary.tokens[i] for i in targets[0, :limit].tolist()]})


@torch.no_grad()
def inspect_generation_pipeline(model, input_ids):
    was_training = model.training
    model.eval()
    try:
        input_ids = input_ids.to(next(model.parameters()).device)
        embeddings = model.embedding(input_ids) * math.sqrt(model.config["d_model"])
        positional = model.position(embeddings)
        memory = positional.new_zeros(positional.size(0), 1, positional.size(2))
        mask = causal_mask(input_ids.size(1), input_ids.device)
        decoded = model.decoder(positional, memory, tgt_mask=mask,
                                tgt_key_padding_mask=input_ids.eq(PAD))
        logits = model.projection(decoded)
        torch.testing.assert_close(logits, model(input_ids))
        stages = dict(input_ids=input_ids, embedding=embeddings, positional_encoding=positional,
                      memory=memory, causal_mask=mask, decoder_output=decoded, logits=logits)
        return pd.DataFrame([{"stage": name, "shape": tuple(value.shape), "dtype": str(value.dtype)}
                             for name, value in stages.items()]).set_index("stage")
    finally:
        model.train(was_training)


@torch.no_grad()
def verify_causal_invariance(model, first, changed_suffix, prefix_length, atol=1e-6, rtol=1e-5):
    """Assert that changed, deleted and PAD-filled suffixes cannot affect the prefix."""
    if not 0 < prefix_length < first.size(1) or first.shape != changed_suffix.shape:
        raise ValueError("Expected equal shapes with a nonempty prefix and suffix")
    torch.testing.assert_close(first[:, :prefix_length], changed_suffix[:, :prefix_length])
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        first, changed_suffix = first.to(device), changed_suffix.to(device)
        padded = first.clone()
        padded[:, prefix_length:] = PAD
        expected = model(first)[:, :prefix_length]
        rows = []
        variants = {"changed_suffix": changed_suffix, "deleted_suffix": first[:, :prefix_length],
                    "padded_suffix": padded}
        for name, inputs in variants.items():
            actual = model(inputs)[:, :prefix_length]
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            rows.append({"check": name, "max_absolute_difference": (actual - expected).abs().max().item(),
                         "passed": True})
        return pd.DataFrame(rows).set_index("check")
    finally:
        model.train(was_training)
