"""Transparent regex tokenizer; no external models or downloads."""
import re
from collections import Counter

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


class Vocabulary:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        if self.tokens[:4] != SPECIALS or len(set(tokens)) != len(tokens):
            raise ValueError("Invalid vocabulary specials or duplicate entries")
        self.indices = {token: index for index, token in enumerate(tokens)}

    @classmethod
    def build(cls, training_texts, max_size=8000, min_frequency=2):
        counts = Counter(token for text in training_texts for token in tokenize(text))
        ordered = sorted(counts, key=lambda token: (-counts[token], token))
        return cls(SPECIALS + [t for t in ordered if counts[t] >= min_frequency
                               and t not in SPECIALS][:max_size - 4])

    def encode(self, text):
        return [self.indices.get(token, UNK) for token in tokenize(text)]

    def decode(self, ids):
        return " ".join(self.tokens[i] for i in ids if i not in (PAD, BOS, EOS))

    def __len__(self):
        return len(self.tokens)
