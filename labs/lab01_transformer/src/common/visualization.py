from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from .text import tokenize


def save_figure(figure, path=None):
    """Finalize and optionally save a figure; display explicitly to avoid double rendering."""
    figure.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return figure


def plot_history(history, task, output_path=None):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    history.plot(x="epoch", y=["train_loss", "validation_loss"], ax=axes[0])
    columns = ["validation_accuracy", "validation_f1"] if task == "classification" else ["validation_perplexity"]
    history.plot(x="epoch", y=columns, ax=axes[1])
    for axis, title, ylabel in zip(axes, ["Функція втрат", "Валідаційні метрики"],
                                   ["Cross-entropy", "Score" if task == "classification" else "Perplexity"]):
        axis.set(title=title, xlabel="Епоха", ylabel=ylabel)
        axis.grid(alpha=.3)
    return save_figure(figure, output_path)


def plot_comparison(table, metric, output_path=None):
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(table.d_model.astype(str) + "/" + table.nhead.astype(str), table[metric])
    axis.set(title="Порівняння на validation", xlabel="d_model / nhead", ylabel=metric)
    axis.grid(axis="y", alpha=.3)
    return save_figure(figure, output_path)


def plot_dataset_overview(data, output_path=None, include_classes=True):
    """Class balance and token lengths, or corpus lengths alone for generation."""
    figure, axes = plt.subplots(1, 2 if include_classes else 1,
                                figsize=(11, 4) if include_classes else (8, 4), squeeze=False)
    axes = axes[0]
    if include_classes:
        data.Category.value_counts().plot.bar(ax=axes[0], rot=0)
        axes[0].set(title="Класи після очищення", xlabel="Клас", ylabel="Кількість")
    axes[-1].hist(data.Message.map(lambda text: len(tokenize(text))), bins=35,
                  edgecolor="black", alpha=.75)
    axes[-1].set(title="Довжини повідомлень", xlabel="Токени", ylabel="Кількість")
    for axis in axes:
        axis.grid(axis="y", alpha=.3)
    return save_figure(figure, output_path)


def plot_positional_encoding(position, output_path=None, positions=32, dimensions=32):
    figure, axis = plt.subplots(figsize=(9, 4))
    values = position.values[0, :positions, :dimensions].detach().cpu().numpy()
    image = axis.imshow(values, aspect="auto", cmap="coolwarm")
    axis.set(title="Синусоїдальне позиційне кодування", xlabel="Вимір embedding", ylabel="Позиція токена")
    figure.colorbar(image, ax=axis, label="PE")
    return save_figure(figure, output_path)


def plot_causal_mask(mask, output_path=None):
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(mask.detach().cpu().int().numpy(), cmap="Greys", vmin=0, vmax=1)
    axis.set(title="Causal mask: 1 = заборонено", xlabel="Key position", ylabel="Query position")
    figure.colorbar(image, ax=axis, ticks=[0, 1])
    return save_figure(figure, output_path)


def plot_confusion_matrix(predictions, output_path=None):
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(predictions.actual, predictions.prediction,
        labels=[0, 1], display_labels=["ham", "spam"], ax=axis)
    axis.set(title="Матриця помилок: test", xlabel="Передбачений клас", ylabel="Справжній клас")
    return save_figure(figure, output_path)
