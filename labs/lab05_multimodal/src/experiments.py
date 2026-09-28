"""Controlled experiment matrix; duplicated conditions share one inference."""
from dataclasses import dataclass
import json
from .config import (QUESTIONS_PATH, GenerationConfig, TEMPERATURE_VALUES,
                     MAX_TOKEN_VALUES, SAMPLING_SEED)


def load_questions():
    return json.loads(QUESTIONS_PATH.read_text(encoding='utf-8'))


@dataclass(frozen=True)
class Case:
    experiment: str
    case_id: str
    question: dict
    generation: GenerationConfig


def experiment_plan():
    questions = load_questions()
    by_id = {q['question_id']: q for q in questions}
    # A real factual baseline case is the smoke test and includes warm-up.
    ordered = [by_id['fact_01']] + [q for q in questions if q['question_id'] != 'fact_01']
    plan = [Case('baseline', q['question_id'], q, GenerationConfig()) for q in ordered]
    plan += [Case('temperature', f'temp_{t}', by_id['analysis_01'],
                  GenerationConfig(True, 64, t, 0.9, SAMPLING_SEED)) for t in TEMPERATURE_VALUES]
    plan += [Case('max_tokens', f'tokens_{n}', by_id['desc_01'],
                  GenerationConfig(max_new_tokens=n)) for n in MAX_TOKEN_VALUES]
    paraphrases = [by_id['fact_01']['question'],
                  'Where is the cabin located horizontally in the image: left, center, or right?',
                  'Which horizontal part of the image contains the cabin: left, center, or right?']
    for i, text in enumerate(paraphrases):
        q = dict(by_id['fact_01'], question=text)
        plan.append(Case('paraphrase', f'phrasing_{i + 1}', q, GenerationConfig()))
    return plan
