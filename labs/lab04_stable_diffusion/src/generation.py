"""Persist each result and reuse only verified, fully matching artifacts."""
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import time
from PIL import Image
from .config import LAB_ROOT


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def identity(spec, context):
    keys = ('model_id', 'revision', 'pipeline_class', 'scheduler', 'scheduler_config',
            'dtype', 'device_strategy', 'torch_version', 'diffusers_version', 'cuda_version', 'gpu_name')
    return {**spec.controls(), **{k: context[k] for k in keys}}


def artifact_id(spec, context):
    return hashlib.sha256(json.dumps(identity(spec, context), sort_keys=True).encode()).hexdigest()[:16]


def filename(spec, key):
    label = spec.style or spec.experiment
    label = re.sub(r'[^A-Za-z0-9_-]', '_', label)
    cfg = str(float(spec.guidance_scale)).replace('.', 'p')
    return f'{label}_seed{spec.seed}_steps{spec.num_inference_steps}_cfg{cfg}_{key}.png'


def cached_result(spec, context, root=LAB_ROOT):
    key = artifact_id(spec, context)
    sidecar = root / 'outputs/metadata' / f'{key}.json'
    if not sidecar.exists():
        return None
    try:
        record = json.loads(sidecar.read_text(encoding='utf-8'))
        if record['identity'] != identity(spec, context):
            return None
        path = root / record['image_path']
        if not path.resolve().is_relative_to(root.resolve()):
            return None
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['image_sha256']:
            return None
        with Image.open(path) as im:
            im.load()
            if im.size != (spec.width, spec.height):
                return None
        return record
    except (OSError, ValueError, KeyError):
        return None


def generate_image(pipe, spec, context, root=LAB_ROOT):
    cached = cached_result(spec, context, root)
    if cached is not None:
        print(f'Reuse {spec.case_id}: {cached["artifact_id"]}', flush=True)
        return cached
    if pipe is None:
        raise RuntimeError('Missing valid artifact: load the pipeline before generation')
    import torch
    key = artifact_id(spec, context)
    path = root / 'outputs/images' / spec.experiment / filename(spec, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device='cpu').manual_seed(spec.seed)
    cuda = context['cuda_available']
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            result = pipe(
                prompt=spec.prompt, negative_prompt=spec.negative_prompt,
                num_inference_steps=spec.num_inference_steps, guidance_scale=spec.guidance_scale,
                width=spec.width, height=spec.height, generator=generator)
        if cuda:
            torch.cuda.synchronize()
    except Exception as exc:
        save_json(root / 'outputs/metadata' / f'failure_{key}.json',
                  {'spec': asdict(spec), 'error_type': type(exc).__name__, 'error': str(exc)})
        raise
    elapsed = time.perf_counter() - start
    image = result.images[0]
    temp = path.with_suffix('.tmp')
    image.save(temp, format='PNG')
    temp.replace(path)
    record = {**context, **asdict(spec), 'artifact_id': key, 'identity': identity(spec, context),
              'image_path': path.relative_to(root).as_posix(),
              'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'generation_time_seconds': elapsed,
              'created_at': datetime.now(timezone.utc).isoformat(),
              'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1024**2 if cuda else None,
              'nsfw_content_detected': result.nsfw_content_detected}
    save_json(root / 'outputs/metadata' / f'{key}.json', record)
    print(f'Generated {spec.case_id}: {elapsed:.2f} s', flush=True)
    return record


def audit_and_manifest(specs, context, root=LAB_ROOT):
    records = [cached_result(s, context, root) for s in specs]
    if any(r is None for r in records):
        raise ValueError('Missing, mismatched or damaged artifact')
    if len({r['artifact_id'] for r in records}) != len(specs):
        raise ValueError('Duplicate generation configurations')
    required = {'artifact_id', 'experiment', 'case_id', 'image_path', 'model_id', 'pipeline_class',
                'prompt', 'negative_prompt', 'seed', 'num_inference_steps', 'guidance_scale',
                'scheduler', 'width', 'height', 'device_strategy', 'dtype', 'python_version',
                'torch_version', 'diffusers_version', 'cuda_version', 'gpu_name',
                'generation_time_seconds', 'created_at'}
    for spec, record in zip(specs, records):
        if not required <= record.keys() or any(record[k] != v for k, v in asdict(spec).items()):
            raise ValueError('Incomplete or inconsistent metadata')
        if record['generation_time_seconds'] <= 0 or any(record.get('nsfw_content_detected') or []):
            raise ValueError('Invalid inference timing or safety-filtered experimental image')
    path = root / 'outputs/metadata/generations.csv'
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        for r in records:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    return records
