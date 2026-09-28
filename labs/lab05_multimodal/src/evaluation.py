"""Join explicitly reviewed answers; never infer subjective scores heuristically."""
import json
import pandas as pd
from .config import LAB_ROOT, OUTPUT_ROOT


def evaluated_results(rows):
    reviews = json.loads((LAB_ROOT / 'report/manual_evaluation.json').read_text(encoding='utf-8'))
    review_map = {r['run_id']: r for r in reviews}
    records = []
    for row in rows:
        review = review_map.get(row['run_id'])
        if review and (review['answer'] != row['answer'] or review['image_sha256'] != row['image_sha256']):
            raise ValueError('Manual evaluation does not match the actual answer/image')
        records.append(dict(row, evaluation_label=review['evaluation_label'] if review else 'NOT_REVIEWED',
                            grounding_issue=review['grounding_issue'] if review else '',
                            notes=review['notes'] if review else 'Потрібне ручне зіставлення з пікселями.'))
    frame = pd.DataFrame(records)
    frame[['run_id', 'question_id', 'evaluation_label', 'grounding_issue', 'notes']].drop_duplicates().to_csv(
        OUTPUT_ROOT / 'metadata/evaluation.csv', index=False, encoding='utf-8-sig')
    return frame


def factual_summary(frame):
    factual = frame[(frame.experiment == 'baseline') & (frame.question_type == 'factual')]
    return factual.evaluation_label.value_counts().reindex(
        ['CORRECT', 'PARTIALLY_CORRECT', 'INCORRECT', 'NOT_REVIEWED'], fill_value=0).rename('count').to_frame()


def table(frame, experiment):
    columns = {
        'baseline': ['question_id', 'question_type', 'question', 'answer', 'evaluation_label', 'runtime_seconds'],
        'temperature': ['temperature', 'answer', 'runtime_seconds', 'generated_token_count', 'evaluation_label'],
        'max_tokens': ['max_new_tokens', 'answer', 'generated_token_count', 'runtime_seconds', 'reached_token_limit'],
        'paraphrase': ['case_id', 'question', 'answer', 'evaluation_label', 'notes'],
    }
    return frame.loc[frame.experiment == experiment, columns[experiment]].reset_index(drop=True)


def markdown_table(frame):
    def escape(value):
        if isinstance(value, float):
            return f'{value:.3f}'
        return str(value).replace('|', '\\|').replace('\n', '<br>')
    rows = ['| ' + ' | '.join(map(escape, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(map(escape, row)) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)
