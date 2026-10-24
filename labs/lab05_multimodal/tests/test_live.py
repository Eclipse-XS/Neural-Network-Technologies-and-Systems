import json
from pathlib import Path

import pytest

from labs.lab05_multimodal.src.config import GROUNDING_INSTRUCTION
from labs.lab05_multimodal.src.live import (
    build_multimodal_message, require_cuda, save_live_result,
)


def test_multimodal_message_contract():
    marker = object()
    messages, prompt = build_multimodal_message(marker, ' Where is the cabin? ')
    assert messages == [{'role': 'user', 'content': [
        {'type': 'image', 'image': marker},
        {'type': 'text', 'text': prompt},
    ]}]
    assert prompt == f'{GROUNDING_INSTRUCTION}\n\nQuestion: Where is the cabin?'
    with pytest.raises(ValueError):
        build_multimodal_message(marker, '  ')


def test_cuda_guard_has_explicit_diagnostic():
    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        cuda = FakeCuda()

    with pytest.raises(RuntimeError, match='Live LLaVA inference was not executed'):
        require_cuda(FakeTorch())


def test_live_output_is_unreviewed_and_isolated(tmp_path):
    evidence = tmp_path / 'evidence' / 'verified_runs.json'
    evidence.parent.mkdir()
    evidence.write_text('immutable', encoding='utf-8')
    before = evidence.read_bytes()
    path = save_live_result({'answer': 'left', 'question': 'Where?'}, tmp_path / 'outputs/live')
    record = json.loads(path.read_text(encoding='utf-8'))
    assert record['review_status'] == 'NOT_REVIEWED'
    assert record['provenance_kind'] == 'live_gpu_inference'
    assert path.parent == tmp_path / 'outputs/live'
    assert evidence.read_bytes() == before
    with pytest.raises(ValueError):
        save_live_result({'answer': ''}, tmp_path / 'outputs/live')
