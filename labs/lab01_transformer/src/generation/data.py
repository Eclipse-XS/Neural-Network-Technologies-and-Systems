import csv
import hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from ..common.text import PAD, BOS, EOS


def inspect_corpus(path):
    """Read strict UTF-8 BBC CSV; categories remain descriptive metadata."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    if not rows or rows[0] != ["category", "text"]:
        raise ValueError("Expected BBC columns: category,text")
    if any(len(row) != 2 for row in rows[1:]):
        raise ValueError("Malformed BBC CSV row")
    frame = pd.DataFrame(rows[1:], columns=rows[0]).replace("", pd.NA)
    return frame


def clean_corpus(frame):
    """Keep the first exact text occurrence without rewriting article contents."""
    if "text" not in frame:
        raise ValueError("Corpus requires a text column")
    clean = frame.dropna(subset=["text"])
    clean = clean.loc[clean.text.str.strip().ne("")].drop_duplicates("text").copy()
    clean["document_id"] = clean.text.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    return clean.reset_index(drop=True)


def validate_splits(splits):
    seen_ids, seen_texts = set(), set()
    for frame in splits.values():
        ids, texts = set(frame.document_id), set(frame.text)
        if len(ids) != len(frame) or len(texts) != len(frame) or seen_ids & ids or seen_texts & texts:
            raise ValueError("Duplicate or overlapping documents across corpus splits")
        seen_ids.update(ids)
        seen_texts.update(texts)


def split_corpus(frame, seed=42):
    train, held_out = train_test_split(frame, test_size=.3, random_state=seed, stratify=frame.category)
    validation, test = train_test_split(held_out, test_size=.5, random_state=seed,
                                      stratify=held_out.category)
    splits = dict(train=train, validation=validation, test=test)
    validate_splits(splits)
    return splits


def corpus_metadata(path, vocabulary, vocab_config, max_len, stride, seed=42):
    return dict(dataset="bbc-news", source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                tokenizer="lowercase_regex_v1", vocabulary_config=dict(vocab_config),
                vocabulary_sha256=hashlib.sha256("\n".join(vocabulary.tokens).encode()).hexdigest(),
                max_len=max_len, stride=stride, split_seed=seed,
                split_policy="exact-text-first; category-stratified 70/15/15")


class LanguageModelDataset(Dataset):
    """Non-overlapping targets; windows never cross document/split boundaries."""
    def __init__(self, texts, vocabulary, max_len=64, document_ids=None, stride=None):
        stride = max_len if stride is None else stride
        if max_len < 1 or stride != max_len:
            raise ValueError("Use positive max_len and stride == max_len for non-overlapping targets")
        texts = list(texts)
        document_ids = list(document_ids) if document_ids is not None else [
            hashlib.sha256(text.encode()).hexdigest() for text in texts]
        if len(document_ids) != len(texts):
            raise ValueError("One document ID is required per text")
        self.samples = []
        self.document_ids = []
        for document_id, text in zip(document_ids, texts):
            tokens = [BOS] + vocabulary.encode(text) + [EOS]
            for start in range(0, len(tokens) - 1, stride):
                window = tokens[start:start + max_len + 1]
                inputs, targets = window[:-1], window[1:]
                inputs = inputs + [PAD] * (max_len - len(inputs))
                targets = targets + [PAD] * (max_len - len(targets))
                self.samples.append((torch.tensor(inputs), torch.tensor(targets)))
                self.document_ids.append(document_id)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def create_loaders(splits, vocabulary, max_len=64, batch_size=32, seed=42, stride=None):
    validate_splits(splits)
    return {name: DataLoader(LanguageModelDataset(frame.text, vocabulary, max_len,
                            document_ids=frame.document_id, stride=stride),
                            batch_size=batch_size, shuffle=name == "train",
                            generator=torch.Generator().manual_seed(seed), num_workers=0)
            for name, frame in splits.items()}
