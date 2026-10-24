"""Small, data-driven figures used by the notebook and report."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

STATUS_COLORS = {'CORRECT': '#277a52', 'INCORRECT': '#a8423d',
                 'HALLUCINATED': '#a8423d', 'HALLUCINATED_DETAIL': '#a66a16'}


def token_budget_figure(frame: pd.DataFrame):
    data = frame.loc[frame.experiment.eq('max_tokens')].sort_values('max_new_tokens')
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout='constrained')
    bars = ax.bar(data.max_new_tokens.astype(str), data.generated_token_count,
                  color=['#a8423d' if value else '#356a8a' for value in data.reached_token_limit])
    for bar, row in zip(bars, data.itertuples()):
        ending = 'ліміт, без EOS' if row.reached_token_limit else 'EOS'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{row.generated_token_count}; {ending}', ha='center', fontsize=9)
    ax.set(title='Бюджет генерації та фактична довжина', xlabel='max_new_tokens, токенів',
           ylabel='Фактично згенеровано токенів', ylim=(0, 64))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
    return fig


def temperature_figure(frame: pd.DataFrame):
    data = frame.loc[frame.experiment.eq('temperature')].sort_values('temperature')
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout='constrained')
    axes[0].bar(data.temperature.astype(str), data.generated_token_count, color='#356a8a')
    axes[0].set(xlabel='temperature', ylabel='Токени відповіді', title='Довжина')
    axes[1].bar(data.temperature.astype(str), data.runtime_seconds, color='#6b7280')
    axes[1].set(xlabel='temperature', ylabel='Історичний час, с', title='Час GPU inference')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Одиничні генерації з seed=42: опис, не статистична оцінка')
    return fig


def error_taxonomy_figure(frame: pd.DataFrame):
    baseline = frame.loc[frame.experiment.eq('baseline')].copy()
    labels = {
        '': 'без зафіксованої помилки',
        'unsupported_roof_material': 'непідтверджений матеріал',
        'spatial_omissions': 'пропущені просторові деталі',
        'missing_evidence': 'відсутній доказ',
        'circular_explanation': 'тавтологічне пояснення',
        'unsupported_numeric_output': 'непідтверджені числа',
    }
    issues = baseline.grounding_issue.map(labels).fillna(baseline.grounding_issue).value_counts()
    fig, ax = plt.subplots(figsize=(8.5, 4.4), layout='constrained')
    colors = ['#356a8a' if label == 'без зафіксованої помилки' else '#a66a16' for label in issues.index]
    ax.barh(issues.index, issues.values, color=colors)
    ax.invert_yaxis()
    ax.set(title='Таксономія ручних зауважень: 10 baseline-відповідей', xlabel='Кількість відповідей')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticks(range(0, int(issues.max()) + 1))
    return fig
