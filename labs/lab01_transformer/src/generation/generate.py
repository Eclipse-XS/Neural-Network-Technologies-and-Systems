import torch
from ..common.text import PAD, UNK, BOS, EOS


def _ban_repeated_ngram(logits, sequence, n):
    if n < 2 or len(sequence) < n - 1:
        return
    prefix = tuple(sequence[-(n - 1):])
    banned = {sequence[index + n - 1] for index in range(len(sequence) - n + 1)
              if tuple(sequence[index:index + n - 1]) == prefix}
    if banned:
        logits[list(banned)] = -torch.inf


def _filter_logits(logits, top_k=None, top_p=None):
    if top_k is not None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < threshold] = -torch.inf
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_indices[remove]] = -torch.inf
    return logits


@torch.no_grad()
def generate_ids(model, context_ids, max_new_tokens=20, *, strategy="greedy",
                 temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0,
                 no_repeat_ngram_size=0, forbid_unk=False, seed=42):
    """Generate one continuation with a rolling context and reproducible sampling.

    Defaults preserve the original greedy decoder. Sampling controls are independent
    so validation ablations can isolate temperature, truncation and repetition rules.
    """
    if strategy not in ("greedy", "sample"):
        raise ValueError("strategy must be greedy or sample")
    if max_new_tokens < 1 or temperature <= 0 or repetition_penalty < 1:
        raise ValueError("Invalid decoding configuration")
    model.eval()
    device = next(model.parameters()).device
    sequence = list(context_ids) or [BOS]
    continuation = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(max_new_tokens):
        context = sequence[-model.config["max_len"]:]
        logits = model(torch.tensor([context], device=device))[0, -1].clone()
        logits[[PAD, BOS]] = -torch.inf
        if forbid_unk:
            logits[UNK] = -torch.inf
        if repetition_penalty > 1:
            for token_id in set(sequence):
                if torch.isfinite(logits[token_id]):
                    logits[token_id] = (logits[token_id] / repetition_penalty
                                        if logits[token_id] > 0
                                        else logits[token_id] * repetition_penalty)
        _ban_repeated_ngram(logits, sequence, no_repeat_ngram_size)
        if strategy == "greedy":
            token = logits.argmax().item()
        else:
            logits = _filter_logits(logits / temperature, top_k, top_p)
            probabilities = torch.softmax(logits, dim=-1)
            token = torch.multinomial(probabilities, 1, generator=generator).item()
        if token == EOS:
            break
        continuation.append(token)
        sequence.append(token)
    return continuation


def generate_text(model, vocabulary, prompt, max_new_tokens=20, **decoding):
    return vocabulary.decode(generate_ids(
        model, [BOS] + vocabulary.encode(prompt), max_new_tokens, **decoding))
