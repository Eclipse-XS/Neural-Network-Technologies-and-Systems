"""BBC corpus EDA and document/window provenance, without model evaluation."""
import matplotlib.pyplot as plt
import pandas as pd
from ..common.text import tokenize, UNK
from ..common.visualization import save_figure
from .data import validate_splits


def summarize_corpus(raw, clean):
    counts = pd.Series(dict(raw_documents=len(raw), duplicate_texts=int(raw.text.duplicated().sum()),
                            missing_texts=int(raw.text.isna().sum()),
                            missing_categories=int(raw.category.isna().sum()), clean_documents=len(clean)))
    lengths = pd.DataFrame({"words": raw.text.dropna().str.split().str.len(),
                           "tokens": raw.text.dropna().map(lambda text: len(tokenize(text)))})
    return counts, lengths.describe(percentiles=[.5, .9, .95, .99]), raw.category.value_counts()


def summarize_corpus_splits(splits, vocabulary, max_len):
    validate_splits(splits)
    rows = []
    for name, frame in splits.items():
        encoded = [vocabulary.encode(text) for text in frame.text]
        token_count = sum(map(len, encoded))
        rows.append(dict(split=name, documents=len(frame), tokens=token_count,
                         windows=sum((len(ids) + 1 + max_len - 1) // max_len for ids in encoded),
                         unk_percent=100 * sum(ids.count(UNK) for ids in encoded) / token_count))
    return pd.DataFrame(rows).set_index("split")


def plot_corpus_overview(frame, output_path=None):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    frame.category.value_counts().plot.bar(ax=axes[0], title="BBC: категорії (лише metadata)")
    axes[0].set(xlabel="Категорія", ylabel="Статті")
    axes[1].hist(frame.text.map(lambda text: len(tokenize(text))), bins=40)
    axes[1].set(title="Довжини статей BBC", xlabel="Токени", ylabel="Статті")
    return save_figure(figure, output_path)
