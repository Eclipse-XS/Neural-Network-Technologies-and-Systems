"""Small contracts shared by the live notebook and CPU-only tests."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import GROUNDING_INSTRUCTION, OUTPUT_ROOT
from .inference import save_json

LIVE_OUTPUT_ROOT = OUTPUT_ROOT / 'live'


def build_multimodal_message(image: Any, question: str) -> tuple[list[dict[str, Any]], str]:
    """Build the structured image/text chat payload accepted by the HF pipeline."""
    clean_question = question.strip()
    if not clean_question:
        raise ValueError('question must not be empty')
    prompt = f'{GROUNDING_INSTRUCTION}\n\nQuestion: {clean_question}'
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image},
        {'type': 'text', 'text': prompt},
    ]}]
    return messages, prompt


def require_cuda(torch_module: Any) -> None:
    """Stop the GPU practical before model loading when CUDA is unavailable."""
    if not torch_module.cuda.is_available():
        raise RuntimeError(
            'CUDA is unavailable. Live LLaVA inference was not executed. '
            'Use notebook 01 for CPU analysis or select a CUDA-enabled kernel.'
        )


def save_live_result(record: dict[str, Any], root: Path = LIVE_OUTPUT_ROOT) -> Path:
    """Persist a new unreviewed run outside immutable historical evidence."""
    answer = record.get('answer')
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError('A live result requires a non-empty answer')
    payload = dict(record)
    payload['review_status'] = 'NOT_REVIEWED'
    payload['provenance_kind'] = 'live_gpu_inference'
    payload.setdefault('created_at', datetime.now(timezone.utc).isoformat())
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    payload['live_run_id'] = identity
    path = root / f'{identity}.json'
    save_json(path, payload)
    return path
