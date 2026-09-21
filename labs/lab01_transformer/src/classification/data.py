import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from ..common.text import PAD, UNK, tokenize


def inspect_data(path):
    frame = pd.read_csv(path, encoding="utf-8")
    if set(frame.columns) != {"Category", "Message"}:
        raise ValueError(f"Unexpected columns: {frame.columns.tolist()}")
    lengths = frame.Message.fillna("").map(lambda text: len(tokenize(text)))
    summary = {
        "rows": len(frame), "missing": frame.isna().sum().to_dict(),
        "duplicate_texts": int(frame.Message.duplicated().sum()),
        "class_counts": frame.Category.value_counts().to_dict(),
        "class_percent": (100 * frame.Category.value_counts(normalize=True)).to_dict(),
        "lengths": lengths.describe(percentiles=[.5, .9, .95, .99]).to_dict(),
    }
    return frame, summary


def clean_data(frame):
    frame = frame.dropna(subset=["Category", "Message"]).copy()
    if not frame.Category.isin(["ham", "spam"]).all():
        raise ValueError("Expected ham/spam labels")
    frame["key"] = frame.Message.map(lambda text: " ".join(tokenize(text)))
    frame = frame[frame.key.ne("")]
    conflicts = frame.groupby("key").Category.nunique()
    # Exclude all instances of contradictory labels rather than choose one label.
    frame = frame[~frame.key.isin(conflicts[conflicts > 1].index)]
    frame = frame.drop_duplicates("key").copy()
    frame["label"] = frame.Category.map({"ham": 0, "spam": 1})
    return frame


def split_data(frame, seed=42):
    train, held_out = train_test_split(frame, test_size=.30, random_state=seed,
                                       stratify=frame.label)
    validation, test = train_test_split(held_out, test_size=.50, random_state=seed,
                                        stratify=held_out.label)
    return {"train": train, "validation": validation, "test": test}


def choose_max_length(training_texts, cap=96):
    lengths = [len(tokenize(text)) for text in training_texts]
    return max(8, min(cap, int(np.ceil(np.quantile(lengths, .95)))))


def class_weights(training_labels):
    counts = np.bincount(training_labels, minlength=2)
    if np.any(counts == 0):
        raise ValueError("Both classes must occur in train")
    return torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)


class SpamDataset(Dataset):
    def __init__(self, frame, vocabulary, max_len):
        self.samples = [torch.tensor(vocabulary.encode(text)[:max_len] or [UNK])
                        for text in frame.Message]
        self.labels = frame.label.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


def collate_messages(batch):
    inputs, labels = zip(*batch)
    input_ids = pad_sequence(inputs, batch_first=True, padding_value=PAD)
    return input_ids, torch.tensor(labels)


def create_loaders(splits, vocabulary, max_len, batch_size=32, seed=42):
    return {name: DataLoader(SpamDataset(frame, vocabulary, max_len),
                            batch_size=batch_size, shuffle=name == "train",
                            generator=torch.Generator().manual_seed(seed),
                            collate_fn=collate_messages, num_workers=0)
            for name, frame in splits.items()}
