from dataclasses import replace
from .config import GenerationSpec, NEGATIVE_PROMPT, STYLES, STYLE_SCENE


def experiment_groups():
    base = GenerationSpec()
    seeds = [base] + [replace(base, experiment='seed', case_id=f'seed_{s}', seed=s) for s in (123, 777, 2026)]
    steps = [replace(base, experiment='steps', case_id=f'steps_{s}', num_inference_steps=s)
             if s != 30 else base for s in (20, 30, 50)]
    cfg = [replace(base, experiment='cfg', case_id=f'cfg_{g}', guidance_scale=g)
           if g != 7.5 else base for g in (5.0, 7.5, 10.0)]
    groups = {'baseline': [base], 'seed': seeds, 'steps': steps, 'cfg': cfg}
    for s, control in zip((42, 123), seeds[:2]):
        groups[f'negative_prompt_seed{s}'] = [control, replace(
            base, experiment='negative_prompt', case_id=f'negative_{s}', seed=s, negative_prompt=NEGATIVE_PROMPT)]
    groups['styles'] = [replace(base, experiment='styles', case_id=f'style_{style}',
                               prompt=f'{STYLE_SCENE}, {suffix}', style=style)
                        for style, suffix in STYLES.items()]
    return groups


def unique_plan():
    seen, result = set(), []
    for group in experiment_groups().values():
        for spec in group:
            key = tuple(spec.controls().items())
            if key not in seen:
                seen.add(key)
                result.append(spec)
    return result
