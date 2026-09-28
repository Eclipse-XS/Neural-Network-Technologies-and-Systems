from dataclasses import replace
import pytest
from labs.lab04_stable_diffusion.src.config import GenerationSpec, STYLE_SCENE, STYLES
from labs.lab04_stable_diffusion.src.experiments import experiment_groups, unique_plan


@pytest.mark.parametrize('group,field,values', [
    ('seed', 'seed', [42, 123, 777, 2026]),
    ('steps', 'num_inference_steps', [20, 30, 50]),
    ('cfg', 'guidance_scale', [5., 7.5, 10.]),
])
def test_one_factor(group, field, values):
    cases = experiment_groups()[group]
    assert [getattr(s, field) for s in cases] == values
    controls = [{k: v for k, v in s.controls().items() if k != field} for s in cases]
    assert all(c == controls[0] for c in controls)


def test_baseline_reuse_and_unique_plan():
    base = GenerationSpec()
    assert (base.seed, base.num_inference_steps, base.guidance_scale, base.width, base.height, base.negative_prompt) == (42, 30, 7.5, 512, 512, None)
    plan = unique_plan()
    assert len(plan) == len({tuple(s.controls().items()) for s in plan}) == 13
    groups = experiment_groups()
    assert groups['seed'][0] is groups['steps'][1] is groups['cfg'][1] is groups['baseline'][0]


def test_negative_pairs_and_styles():
    groups = experiment_groups()
    for seed in (42, 123):
        a, b = groups[f'negative_prompt_seed{seed}']
        assert a.seed == b.seed == seed
        assert a.negative_prompt is None and b.negative_prompt
        assert {k for k in a.controls() if a.controls()[k] != b.controls()[k]} == {'negative_prompt'}
    for s in groups['styles']:
        assert s.prompt == f'{STYLE_SCENE}, {STYLES[s.style]}'
        assert {k: v for k, v in s.controls().items() if k != 'prompt'} == {
            k: v for k, v in GenerationSpec().controls().items() if k != 'prompt'}


@pytest.mark.parametrize('kwargs', [{'seed': -1}, {'width': 511}, {'num_inference_steps': 0},
                                  {'guidance_scale': float('nan')}, {'prompt': ' '}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        replace(GenerationSpec(), **kwargs)
