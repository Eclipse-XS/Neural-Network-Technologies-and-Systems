import torch
from ..common.text import PAD, BOS, EOS


@torch.no_grad()
def generate_ids(model, context_ids, max_new_tokens=20):
    """Return only the greedy continuation, with a rolling context window."""
    model.eval()
    device = next(model.parameters()).device
    sequence = list(context_ids) or [BOS]
    continuation = []
    for _ in range(max_new_tokens):
        context = sequence[-model.config["max_len"]:]
        logits = model(torch.tensor([context], device=device))[0, -1].clone()
        logits[[PAD, BOS]] = -torch.inf
        token = logits.argmax().item()
        if token == EOS:
            break
        continuation.append(token)
        sequence.append(token)
    return continuation


def generate_text(model, vocabulary, prompt, max_new_tokens=20):
    return vocabulary.decode(generate_ids(model, [BOS] + vocabulary.encode(prompt), max_new_tokens))
