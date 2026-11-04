"""Unit tests for tool provenance and deterministic recovery (DEFECT-03)."""
import pytest
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import ScoreType, ToolCallOrigin
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, EvidenceChunk, Score
from labs.lab07_final_project.src.knowledge_copilot.agents.recovery import (
    attempt_arithmetic_recovery,
    has_arithmetic_intent,
)


def make_evidence_chunk(doc_id: str, chunk_id: str, text: str) -> EvidenceChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        ordinal=0,
        text=text,
        section="Results",
        token_count=len(text.split()),
        sha256="hash123",
    )
    score = Score(
        retrieval_method="hybrid",
        score_type=ScoreType.RRF_SCORE,
        score_value=0.033,
        ranking_position=1,
    )
    return EvidenceChunk(chunk=chunk, score=score)


def test_arithmetic_intent_detection():
    assert has_arithmetic_intent("Calculate the difference between Lab 4 and Lab 5 runs")
    assert has_arithmetic_intent("Яка різниця між кількістю генерацій у Лаб 4 і Лаб 5?")
    assert has_arithmetic_intent("What is the sum of unique images?")
    assert not has_arithmetic_intent("What model was used in Lab 6?")


def test_arithmetic_recovery_tags_origin_strictly_as_recovery_executed():
    """DEFECT-03: Controller-executed arithmetic must NEVER be reported as MODEL_REQUESTED."""
    ev4 = make_evidence_chunk("lab04", "chunk-04", "У Лабораторній 4 виконано 17 унікальних генерацій.")
    ev5 = make_evidence_chunk("lab05", "chunk-05", "У Лабораторній 5 виконано 13 унікальних запусків.")
    
    user_query = "Calculate the difference between Lab 4 and Lab 5 unique executions."
    draft_answer = "Lab 4 had 17 executions and Lab 5 had 13 executions."
    
    recovered_text, tool_record = attempt_arithmetic_recovery(
        user_input=user_query,
        draft_answer=draft_answer,
        evidence=[ev4, ev5],
    )
    
    assert recovered_text is not None
    assert tool_record is not None
    assert "4" in recovered_text  # 17 - 13 = 4
    assert tool_record.tool_name == "calculate"
    assert tool_record.arguments == {"expression": "17 - 13"}
    assert tool_record.result == {"expression": "17 - 13", "result": 4.0}
    # Strict assertion for DEFECT-03:
    assert tool_record.origin == ToolCallOrigin.RECOVERY_EXECUTED
    assert tool_record.origin != ToolCallOrigin.MODEL_REQUESTED
