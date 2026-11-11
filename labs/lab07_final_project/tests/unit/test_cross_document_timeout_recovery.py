"""Regression tests for cross-document multi-step execution and timeout resilience."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus, ScoreType
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, EvidenceChunk, Score
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.llm_client import LocalLLMClient
from labs.lab07_final_project.src.knowledge_copilot.agents.orchestrator import AgentOrchestrator
from labs.lab07_final_project.src.knowledge_copilot.agents.recovery import (
    has_arithmetic_intent,
    attempt_arithmetic_recovery,
)


def make_chunk(doc_id: str, chunk_id: str, text: str) -> EvidenceChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        ordinal=0,
        text=text,
        section="Section",
        token_count=len(text.split()),
        sha256="dummy_sha",
    )
    score = Score(
        retrieval_method="hybrid",
        score_type=ScoreType.RRF_SCORE,
        score_value=0.033,
        ranking_position=1,
    )
    return EvidenceChunk(chunk=chunk, score=score)


def test_timeout_configurations_resilient_to_180s():
    """Verify that timeout thresholds have been raised to 180.0s to prevent 120.0s agent loop aborts."""
    client = LocalLLMClient(base_url="http://localhost:1234/v1")
    assert client.timeout_seconds == 180.0

    orchestrator = AgentOrchestrator(
        llm_client=client,
        search_fn=lambda **kw: [],
        list_sources_fn=lambda: [],
    )
    assert orchestrator.timeout_seconds == 180.0


def test_arithmetic_intent_comprehensive():
    """Verify arithmetic intent detection covers both Ukrainian and English terms."""
    queries_positive = [
        "Обчисліть різницю між оновленим лімітом пам'яті Redis та початковим дефолтним лімітом",
        "Обчисліть скільки товарів залишається після вилучення 17 довгих записів",
        "Обчисліть тривалість інциденту Redis OOM у хвилинах",
        "Обчисліть добуток кількості шарів енкодера зору SigLIP та мовної частини Qwen2",
        "Calculate the difference between the full multimodal parameter count and backbone",
        "Compute the difference between Hit@5 and Recall@5",
        "Calculate the remaining free VRAM when 3.23 GiB is available out of 4.000 GiB",
    ]
    for q in queries_positive:
        assert has_arithmetic_intent(q), f"Failed to detect arithmetic intent in: {q}"

    queries_negative = [
        "What vector database was chosen in ADR 001?",
        "На яких портах працюють FastAPI та Streamlit?",
        "How is document parsing resilience maintained?",
    ]
    for q in queries_negative:
        assert not has_arithmetic_intent(q), f"False positive arithmetic intent in: {q}"


def test_arithmetic_recovery_single_and_cross_document():
    """Verify deterministic arithmetic recovery works when operands are in a single document or question."""
    # 1. Operands in question (8501 and 8000 -> 501)
    ev = [make_chunk("adr", "c1", "FastAPI runs on port 8000, Streamlit runs on port 8501.")]
    q = "Обчисліть різницю між портом вебінтерфейсу Streamlit (8501) та портом бекенду FastAPI (8000)."
    ans = "Різниця становить 501."
    recovered_text, tool_record = attempt_arithmetic_recovery(q, ans, ev)
    assert tool_record is not None
    assert tool_record.arguments["expression"] == "8501 - 8000"
    assert tool_record.result["result"] == 501.0

    # 2. Operands in evidence (185 and 17 -> 168)
    ev2 = [make_chunk("lab03", "c2", "Було завантажено 185 товарів, з яких 17 було обрізано.")]
    q2 = "Обчисліть скільки товарів залишається після вилучення 17 довгих записів зі 185 товарів."
    ans2 = "Залишається 168 товарів."
    recovered_text2, tool_record2 = attempt_arithmetic_recovery(q2, ans2, ev2)
    assert tool_record2 is not None
    assert tool_record2.arguments["expression"] == "185 - 17"
    assert tool_record2.result["result"] == 168.0

    # 3. Product multiplication (26 * 24 -> 624)
    ev3 = [make_chunk("lab05", "c3", "SigLIP має 26 шарів, а Qwen2 містить 24 шари.")]
    q3 = "Обчисліть добуток кількості шарів енкодера зору SigLIP (26) та мовної частини Qwen2 (24)."
    ans3 = "Правильна цифра — 624."
    recovered_text3, tool_record3 = attempt_arithmetic_recovery(q3, ans3, ev3)
    assert tool_record3 is not None
    assert tool_record3.arguments["expression"] == "26 * 24"
    assert tool_record3.result["result"] == 624.0
