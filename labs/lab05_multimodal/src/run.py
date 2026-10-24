"""Structured CLI for cache-aware LLaVA experiments."""
import json
from PIL import Image
from .config import INPUT_IMAGE_PATH, OUTPUT_ROOT
from .experiments import experiment_plan
from .inference import answer_question, cached_result, save_json, image_sha, signature
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
    counts = {name: sum(case.experiment == name for case in plan)
              for name in ('baseline', 'temperature', 'max_tokens', 'paraphrase')}
    print('LAB 05 | MULTIMODAL VQA')
    print('\n[1/5] ENVIRONMENT')
    print(f"  Model       : {env.get('gpu_name') or 'CPU'}")
    print(f"  Device      : {'cuda:0' if env['cuda_available'] else 'cpu'}")
    print(f"  Precision   : {'float16' if env['cuda_available'] else 'float32'}")
    print(f"  Input       : {INPUT_IMAGE_PATH.name}")
    print('\n[2/5] EXPERIMENT PLAN')
    print(f"  Baseline    : {counts['baseline']} rows")
    print(f"  Temperature : {counts['temperature']} rows")
    print(f"  Token budget: {counts['max_tokens']} rows")
    print(f"  Paraphrase  : {counts['paraphrase']} rows")
    pipe = None
    cache_complete = all(cached_result(case, context) for case in plan)
    print('\n[3/5] MODEL / CACHE')
    if not cache_complete:
        if not env['cuda_available']:
            print('  CUDA unavailable. CPU inference is possible but may be slow and memory-intensive.')
        pipe, model_info = load_pipeline()
        save_json(OUTPUT_ROOT / 'metadata/model.json', model_info)
        save_json(OUTPUT_ROOT / 'metadata/environment.json', env)
        print(f"  Loaded      : {model_info['model_class']} on {model_info['device_strategy']}", flush=True)
    else:
        print('  All 17 unique generations are valid cache hits; GPU inference will not run.')
    with Image.open(INPUT_IMAGE_PATH) as source_image:
        image = source_image.convert('RGB')
    print('\n[4/5] EXECUTION')
    rows, unique = [], {}
    for index, case in enumerate(plan, start=1):
        run_id = signature(case, context)
        existing = unique.get(run_id)
        mode = 'REUSED RESULT' if existing else ('CACHE HIT' if cached_result(case, context) else 'INFERENCE')
        record = existing or answer_question(pipe, image, case, context)
        unique[record['run_id']] = record
        rows.append(dict(record, experiment=case.experiment, case_id=case.case_id))
        generation = case.generation
        params = f"max_new_tokens={generation.max_new_tokens}, do_sample={generation.do_sample}"
        if generation.do_sample:
            params += f", temperature={generation.temperature}, top_p={generation.top_p}, seed={generation.seed}"
        print(f"  [{index:02d}/{len(plan):02d}] {case.experiment}/{case.case_id}")
        print(f"    Mode      : {mode}")
        print(f"    Parameters: {params}")
        print(f"    Question  : {case.question['question']}")
        print(f"    Answer    : {record['answer']}")
        print(f"    Tokens    : {record.get('generated_token_count')} | EOS={record.get('ended_with_eos')}")
        if mode == 'INFERENCE':
            print(f"    Time      : {record['runtime_seconds']:.3f} s")
        else:
            print(f"    Time      : not measured ({record['runtime_seconds']:.3f} s belongs to historical inference)")
        # A manifest separates experimental membership from physical generations.
        save_json(OUTPUT_ROOT / 'metadata/manifest.json', rows)
    path = OUTPUT_ROOT / 'responses/responses.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in unique.values()), encoding='utf-8')
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUTPUT_ROOT / 'responses/responses.csv', index=False, encoding='utf-8-sig')
    print('\n[5/5] SUMMARY')
    print(f'  Experiment rows : {len(rows)}')
    print(f'  Unique results  : {len(unique)}')
    print(f"  Execution       : {'cache only' if cache_complete else 'cache plus missing inference'}", flush=True)
    return rows


if __name__ == '__main__':
    run_experiments()
