"""Run from repository root: python -m labs.lab04_stable_diffusion.src.run."""
import json
from .config import LAB_ROOT, ModelConfig
from .experiments import unique_plan
from .generation import cached_result, generate_image, save_json, audit_and_manifest
from .pipeline import load_pipeline, environment
from .visualization import build_figures


def main():
    plan = unique_plan()
    context_path = LAB_ROOT/'outputs/metadata/environment.json'
    pipe = None
    context = json.loads(context_path.read_text(encoding='utf-8')) if context_path.exists() else None
    env = environment()
    matching = context and all(context[k] == env[k] for k in ('versions', 'cuda_version', 'gpu_name'))
    matching = matching and context['revision'] == ModelConfig().revision and context['model_id'] == ModelConfig().model_id
    if not matching or not all(cached_result(s, context) for s in plan):
        pipe, context = load_pipeline()
        save_json(context_path, context)
    for spec in plan:
        generate_image(pipe, spec, context)
    records = audit_and_manifest(plan, context)
    build_figures(context)
    print(f'Audited {len(records)} unique images.', flush=True)


if __name__ == '__main__':
    main()
