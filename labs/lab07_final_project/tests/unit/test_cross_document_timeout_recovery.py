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
    assert client.timeout_seconds >= 180.0

    orchestrator = AgentOrchestrator(
        llm_client=client,
        search_fn=lambda **kw: [],
        list_sources_fn=lambda: [],
    )
    assert orchestrator.timeout_seconds >= 180.0


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

    # 4. HH:MM timestamp duration in minutes (v3_032: 14:20 -> 14:47 = 27)
    ev4 = [make_chunk("postmortem", "c4", "14:20 started, 14:47 restarted service.")]
    q4 = "Обчисліть тривалість інциденту в хвилинах між 14:20 UTC та 14:47 UTC."
    _, tr4 = attempt_arithmetic_recovery(q4, "Тривалість становить 27 хвилин.", ev4)
    assert tr4 is not None
    assert tr4.arguments["expression"] == "887 - 860"
    assert tr4.result["result"] == 27.0

    # 5. Range difference max - min across 3 numbers (v3_037: 24, 64, 128 -> 104)
    ev5 = [make_chunk("lab05", "c5", "Tested max_new_tokens 24, 64, and 128.")]
    q5 = "In Lab 5, token limits 24, 64, and 128 were evaluated for desc_01. What is the difference in tokens between the maximum (128) and minimum (24) limits?"
    _, tr5 = attempt_arithmetic_recovery(q5, "At 64 and 128 tokens...", ev5)
    assert tr5 is not None
    assert tr5.arguments["expression"] == "128 - 24"
    assert tr5.result["result"] == 104.0

    # 6. Unit conversion division (v3_039: 60 req/min -> 1 req/s; v3_040: 5000 ms -> 5 s)
    ev6 = [make_chunk("api", "c6", "Rate limit: 60 requests per minute.")]
    _, tr6 = attempt_arithmetic_recovery("If rate limit is 60 requests per minute, how many requests per second on average?", "1 request per second.", ev6)
    assert tr6 is not None
    assert tr6.arguments["expression"] == "60 / 60"
    assert tr6.result["result"] == 1.0

    ev7 = [make_chunk("adr", "c7", "PRAGMA busy_timeout = 5000.")]
    _, tr7 = attempt_arithmetic_recovery("Convert PRAGMA busy_timeout of 5000 ms into seconds.", "5 seconds.", ev7)
    assert tr7 is not None
    assert tr7.arguments["expression"] == "5000 / 1000"
    assert tr7.result["result"] == 5.0

def test_combined_total_average_and_single_digit_float_recovery():
    """Verify combined total additive intent, arithmetic mean, 4.000 GiB extraction, and operand-scoped citations."""
    # 1. Combined total (EN + UK) across documents with distractor chunk
    assert has_arithmetic_intent("What is the combined total of unique generations in Lab 4 and test cases in Lab 6?")
    assert has_arithmetic_intent("Яка загальна кількість унікальних генерацій у Lab 4 та тестових сценаріїв у Lab 6?")

    ev_l4 = make_chunk("lab04", "c4", "Було виконано 13 унікальних генерацій.")
    ev_l6 = make_chunk("lab06", "c6", "Набір містив 11 тестових сценаріїв.")
    ev_noise = make_chunk("adr001", "c_noise", "Redis runs on port 6379.")

    rec_txt, rec_tr = attempt_arithmetic_recovery(
        "Яка загальна кількість унікальних генерацій у Lab 4 та тестових сценаріїв у Lab 6?",
        "У Lab 4 зафіксовано 13 генерацій, а у Lab 6 — 11 сценаріїв.",
        [ev_l4, ev_l6, ev_noise],
    )
    assert rec_tr is not None
    assert rec_tr.arguments["expression"] == "13 + 11"
    assert rec_tr.result["result"] == 24.0
    assert rec_txt is not None
    assert "[lab04#c4]" in rec_txt and "[lab06#c6]" in rec_txt
    assert "[adr001#c_noise]" not in rec_txt

    # 2. Arithmetic mean vs rate division
    _, avg_tr = attempt_arithmetic_recovery(
        "What is the average of 120 and 80?",
        "The average is 100.",
        [make_chunk("api", "c_avg", "Values 120 and 80 ms.")],
    )
    assert avg_tr is not None
    assert avg_tr.arguments["expression"] == "(120 + 80) / 2"
    assert avg_tr.result["result"] == 100.0

    # 3. 4.000 GiB single-digit float in question after stripping Lab 5
    _, gib_tr = attempt_arithmetic_recovery(
        "Calculate the difference between 4.000 GiB total VRAM and 3.230 GiB free in Lab 5.",
        "Used VRAM is 0.77 GiB.",
        [make_chunk("lab05", "c_gib", "Total VRAM 4.000 GiB, free 3.230 GiB.")],
    )
    assert gib_tr is not None
    assert gib_tr.arguments["expression"] == "4 - 3.23"
    assert abs(gib_tr.result["result"] - 0.77) < 1e-6
