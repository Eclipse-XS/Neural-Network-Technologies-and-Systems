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


STATUS_LIVE = "LIVE_GENERATED"
STATUS_CACHE = "CACHE_REUSED"
STATUS_REPLAY = "HISTORICAL_REPLAY"
STATUS_NOT_EXECUTED = "NOT_EXECUTED"
STATUS_FAILED = "FAILED"


def save_json(path: Path, value: object) -> None:
    """Atomically persist JSON metadata without permitting NaN values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def identity(spec, context):
    """Combine generation controls and runtime provenance into cache identity."""
    keys = ('model_id', 'revision', 'pipeline_class', 'scheduler', 'scheduler_config',
            'dtype', 'device_strategy', 'torch_version', 'diffusers_version', 'cuda_version', 'gpu_name')
    return {**spec.controls(), **{k: context[k] for k in keys}}


def artifact_id(spec, context):
    """Return a stable short hash for one fully specified generation."""
    return hashlib.sha256(json.dumps(identity(spec, context), sort_keys=True).encode()).hexdigest()[:16]


def filename(spec, key):
    """Create a filesystem-safe PNG name containing the primary controls."""
    label = spec.style or spec.experiment
    label = re.sub(r'[^A-Za-z0-9_-]', '_', label)
    cfg = str(float(spec.guidance_scale)).replace('.', 'p')
    return f'{label}_seed{spec.seed}_steps{spec.num_inference_steps}_cfg{cfg}_{key}.png'


def cached_result(spec, context, root=LAB_ROOT):
    """Return metadata only when identity, hash, decoding and dimensions agree."""
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


def generate_image(pipe, spec, context, root=LAB_ROOT, force_regenerate=False, run_id=None):
    """Generate or safely reuse one image and persist its complete provenance.

    The measured duration covers pipeline inference, not model loading. CUDA peak
    memory is recorded when available. Failed inference writes a diagnostic
    sidecar and re-raises the original exception.
    """
    if not force_regenerate:
        cached = cached_result(spec, context, root)
        if cached is not None:
            print(f'Reuse {spec.case_id}: {cached["artifact_id"]}', flush=True)
            return {
                **cached,
                'execution_status': STATUS_CACHE,
                'status': STATUS_CACHE,
                'generated_in_current_run': False,
                'cache_hit': True,
                'pipeline_loaded_now': pipe is not None,
                'run_id': run_id or f'reuse_{cached["artifact_id"]}',
                'verified_at': datetime.now(timezone.utc).isoformat()
            }
    if pipe is None:
        raise RuntimeError('Missing valid artifact: load the pipeline before generation')
    import torch
    key = artifact_id(spec, context)
    path = root / 'outputs/images' / spec.experiment / filename(spec, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device='cpu').manual_seed(spec.seed)
    cuda = context.get('cuda_available', False)
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
                  {'spec': asdict(spec), 'error_type': type(exc).__name__, 'error': str(exc),
                   'execution_status': STATUS_FAILED})
        raise
    elapsed = time.perf_counter() - start
    image = result.images[0]
    temp = path.with_suffix('.tmp')
    image.save(temp, format='PNG')
    temp.replace(path)
    current_run_id = run_id or f'run_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}_{key}'
    record = {
        **context, **asdict(spec), 'artifact_id': key, 'identity': identity(spec, context),
        'image_path': path.relative_to(root).as_posix(),
        'image_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'generation_time_seconds': elapsed,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'verified_at': datetime.now(timezone.utc).isoformat(),
        'peak_cuda_memory_mb': torch.cuda.max_memory_allocated()/1024**2 if cuda else None,
        'nsfw_content_detected': result.nsfw_content_detected,
        'execution_status': STATUS_LIVE,
        'status': STATUS_LIVE,
        'generated_in_current_run': True,
        'pipeline_loaded_now': True,
        'cache_hit': False,
        'run_id': current_run_id
    }
    save_json(root / 'outputs/metadata' / f'{key}.json', record)
    print(f'Generated {spec.case_id}: {elapsed:.2f} s', flush=True)
    return record


def audit_and_manifest(specs, context, root=LAB_ROOT):
    """Validate every planned artifact and write the canonical CSV manifest."""
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
