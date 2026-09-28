"""Run from repository root: python -m labs.lab05_multimodal.src.run."""
import json
from PIL import Image
from .config import INPUT_IMAGE_PATH, OUTPUT_ROOT
from .experiments import experiment_plan
from .inference import answer_question, cached_result, save_json, image_sha
from .model import environment, load_pipeline


def run_experiments():
    env = environment()
    save_json(OUTPUT_ROOT / 'metadata/preflight_latest.json', env)
    source = json.loads(INPUT_IMAGE_PATH.with_name('source.json').read_text(encoding='utf-8'))
    if source['sha256'] != image_sha():
        raise ValueError('Input image differs from its recorded provenance')
    context = {k: env[k] for k in ('python', 'versions', 'cuda_runtime', 'gpu_name')}
    context.update(dtype='torch.float16' if env['cuda_available'] else 'torch.float32',
                   device_strategy='cuda:0' if env['cuda_available'] else 'cpu',
                   attention='sdpa', protocol_version=1)
    plan = experiment_plan()
    pipe = None
    if not all(cached_result(case, context) for case in plan):
        pipe, model_info = load_pipeline()
        save_json(OUTPUT_ROOT / 'metadata/model.json', model_info)
        save_json(OUTPUT_ROOT / 'metadata/environment.json', env)
        print(f"Loaded {model_info['model_class']} on {model_info['device_strategy']}", flush=True)
    with Image.open(INPUT_IMAGE_PATH) as source_image:
        image = source_image.convert('RGB')
    rows, unique = [], {}
    for case in plan:
        record = answer_question(pipe, image, case, context)
        unique[record['run_id']] = record
        rows.append(dict(record, experiment=case.experiment, case_id=case.case_id))
        # A manifest separates experimental membership from physical generations.
        save_json(OUTPUT_ROOT / 'metadata/manifest.json', rows)
    path = OUTPUT_ROOT / 'responses/responses.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in unique.values()), encoding='utf-8')
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / 'responses/responses.csv', index=False, encoding='utf-8-sig')
    print(f'Complete: {len(rows)} experiment rows, {len(unique)} unique generations.', flush=True)
    return rows


if __name__ == '__main__':
    run_experiments()
