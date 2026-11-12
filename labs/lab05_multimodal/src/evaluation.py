"""Join explicitly reviewed answers when available without heuristic scoring."""
import json
import pandas as pd
from .config import OUTPUT_ROOT, REVIEW_LABELS_PATH


def evaluated_results(rows, reviews_path=REVIEW_LABELS_PATH):
    reviews = (json.loads(reviews_path.read_text(encoding='utf-8'))
               if reviews_path is not None and reviews_path.is_file() else [])
    review_map = {r['run_id']: r for r in reviews}
    records = []
    for row in rows:
        review = review_map.get(row['run_id'])
        if review and (review['answer'] != row['answer'] or review['image_sha256'] != row['image_sha256']):
            raise ValueError('Review label does not match the actual answer/image')
        records.append(dict(row, evaluation_label=review['evaluation_label'] if review else 'NOT_REVIEWED',
                            grounding_issue=review['grounding_issue'] if review else '',
                            notes=review['notes'] if review else 'Compare answer against image pixels.'))
    frame = pd.DataFrame(records)
    (OUTPUT_ROOT / 'metadata').mkdir(parents=True, exist_ok=True)
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
