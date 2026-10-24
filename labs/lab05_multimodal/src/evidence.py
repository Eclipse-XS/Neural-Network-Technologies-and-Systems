"""Read and validate the compact, versioned record of the historical GPU run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import (INPUT_IMAGE_PATH, MODEL_ID, MODEL_REVISION,
                     VERIFIED_RUNS_PATH)

REQUIRED_RUN_FIELDS = {
    'run_id', 'question_id', 'question', 'answer', 'model_id', 'revision',
    'image_sha256', 'generation_config', 'generated_token_count',
    'ended_with_eos', 'runtime_seconds', 'execution_context', 'created_at',
}


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 of a file without loading it twice."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_verified_runs(payload: dict[str, Any]) -> None:
    """Reject incomplete or internally inconsistent historical evidence."""
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported verified evidence schema')
    runs = payload.get('runs')
    if not isinstance(runs, list) or len(runs) != 17:
        raise ValueError('Verified evidence must contain 17 unique generations')
    actual_image_sha = file_sha256(INPUT_IMAGE_PATH)
    seen: set[str] = set()
    for index, run in enumerate(runs):
        missing = REQUIRED_RUN_FIELDS - set(run)
        if missing:
            raise ValueError(f'Run {index} is missing fields: {sorted(missing)}')
        if run['run_id'] in seen:
            raise ValueError(f"Duplicate run_id: {run['run_id']}")
        seen.add(run['run_id'])
        if not isinstance(run['answer'], str) or not run['answer'].strip():
            raise ValueError(f'Run {index} has an empty answer')
        if run['model_id'] != MODEL_ID or run['revision'] != MODEL_REVISION:
            raise ValueError(f'Run {index} belongs to another checkpoint')
        if run['image_sha256'] != actual_image_sha:
            raise ValueError(f'Run {index} belongs to another image')
        if not isinstance(run['generation_config'], dict):
            raise ValueError(f'Run {index} has invalid generation_config')


def load_verified_runs(path: Path = VERIFIED_RUNS_PATH) -> list[dict[str, Any]]:
    """Load historical runs after validating schema and image provenance."""
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'Cannot read verified evidence: {path}') from exc
    validate_verified_runs(payload)
    return payload['runs']


def verified_run_map() -> dict[str, dict[str, Any]]:
    """Index verified generations by their content-addressed run id."""
    return {run['run_id']: run for run in load_verified_runs()}


def historical_experiment_rows() -> list[dict[str, Any]]:
    """Expand 17 physical generations into the documented 19 experiment rows."""
    from dataclasses import asdict
    from .experiments import experiment_plan

    runs = load_verified_runs()
    lookup = {(run['question'], tuple(sorted(run['generation_config'].items()))): run
              for run in runs}
    rows = []
    for case in experiment_plan():
        key = (case.question['question'], tuple(sorted(asdict(case.generation).items())))
        if key not in lookup:
            raise ValueError(f'Missing verified result for {case.experiment}/{case.case_id}')
        run = lookup[key]
        generation = run['generation_config']
        rows.append(dict(run, **generation, experiment=case.experiment, case_id=case.case_id))
    if len(rows) != 19 or len({row['run_id'] for row in rows}) != 17:
        raise ValueError('Historical matrix must contain 19 rows and 17 unique runs')
    return rows
