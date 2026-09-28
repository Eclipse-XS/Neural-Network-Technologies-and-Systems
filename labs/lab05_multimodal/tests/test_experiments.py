from dataclasses import asdict, replace
import pytest
from labs.lab05_multimodal.src.config import GenerationConfig
from labs.lab05_multimodal.src.experiments import experiment_plan
from labs.lab05_multimodal.src import inference


def test_controlled_matrix():
    plan = experiment_plan()
    baseline = [c for c in plan if c.experiment == 'baseline']
    assert len(baseline) == 10 and baseline[0].question['question_type'] == 'factual'
    assert all(c.generation.kwargs() == {'do_sample': False, 'max_new_tokens': 64} for c in baseline)
    for experiment, variable, values in [
        ('temperature', 'temperature', [0.2, 0.7, 1.0]),
        ('max_tokens', 'max_new_tokens', [24, 64, 128]),
    ]:
        cases = [c for c in plan if c.experiment == experiment]
        assert [getattr(c.generation, variable) for c in cases] == values
        controls = [{k: v for k, v in asdict(c.generation).items() if k != variable} for c in cases]
        assert controls[0] == controls[1] == controls[2]
        assert len({c.question['question'] for c in cases}) == 1
    para = [c for c in plan if c.experiment == 'paraphrase']
    assert len({c.question['question'] for c in para}) == 3
    assert all(c.generation == GenerationConfig() for c in para)
    assert len({inference.signature(c, {}) for c in plan}) == 17


def test_sampling_validation():
    for kwargs in [dict(temperature=0), dict(top_p=0.9), dict(max_new_tokens=0),
                   dict(do_sample=True), dict(do_sample=True, temperature=float('nan'), top_p=.9, seed=42)]:
        with pytest.raises(ValueError):
            GenerationConfig(**kwargs)


def test_resume_and_invalidation(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, 'OUTPUT_ROOT', tmp_path)
    case = experiment_plan()[0]
    context = {'version': 'test'}
    run_id = inference.signature(case, context)
    record = {'run_id': run_id, 'answer': 'Left.'}
    inference.save_json(tmp_path / f'responses/{run_id}.json', record)
    assert inference.cached_result(case, context) == record
    assert inference.cached_result(case, {'version': 'changed'}) is None
    assert inference.cached_result(replace(case, generation=GenerationConfig(max_new_tokens=24)), context) is None
    assert inference.cached_result(replace(case, question=dict(case.question, question='Changed?')), context) is None
    monkeypatch.setattr(inference, 'GROUNDING_INSTRUCTION', 'Different grounding instruction')
    assert inference.cached_result(case, context) is None
    monkeypatch.undo()


def test_assistant_extraction():
    assert inference.extract_answer([{'generated_text': ' Left. '}]) == 'Left.'
    output = [{'generated_text': [
        {'role': 'user', 'content': 'Question?'},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Left.'}]}]}]
    assert inference.extract_answer(output) == 'Left.'
    for output in [[], [{'generated_text': ''}], [{'generated_text': [{'role': 'user', 'content': 'Q?'}]}]]:
        with pytest.raises(ValueError):
            inference.extract_answer(output)
