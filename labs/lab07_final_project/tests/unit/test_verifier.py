"""Unit tests for semantic grounding and citation verification (DEFECT-02)."""
import pytest
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus, ScoreType
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, EvidenceChunk, Score
from labs.lab07_final_project.src.knowledge_copilot.agents.verifier import (
    extract_citations,
    extract_suspicious_citations,
    is_abstention,
    verify_grounding,
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


def test_extract_citations():
    text = "Lab 4 used 17 unique runs [lab04#chunk-abc1234] while Lab 5 used 13 [lab05#chunk-def5678]."
    cites = extract_citations(text)
    assert cites == ["[lab04#chunk-abc1234]", "[lab05#chunk-def5678]"]


def test_detects_suspicious_placeholder_citations():
    text = "The answer is supported by [source_id#chunk_id] and [lab03_report.md]."
    suspicious = extract_suspicious_citations(text)
    assert "[source_id#chunk_id]" in suspicious
    assert "[lab03_report.md]" in suspicious


def test_refusal_abstention_detection():
    assert is_abstention("I cannot determine the outdoor temperature from the indexed reports.")
    assert is_abstention("На основі звітів не можу визначити погодні умови.")
    assert is_abstention("У наданих документах недостатньо даних для цієї відповіді.")
    assert not is_abstention("Lab 4 generated 17 images.")


def test_valid_citation_attached_to_unsupported_claim_is_rejected():
    """DEFECT-02 test: Citation ID is valid, but the claim asserts 999 which is absent from evidence."""
    ev = make_evidence_chunk(
        doc_id="lab04",
        chunk_id="chunk-abc1234",
        text="Під час виконання Лабораторної роботи №4 було створено 17 унікальних зображень.",
    )
    
    # Claim falsely asserts 999 instead of 17, but cites the real chunk
    answer = "Lab 4 created 999 unique images [lab04#chunk-abc1234]."
    status, cited, invalid, claims = verify_grounding(answer, [ev])
    
    # Must NOT be marked GROUNDED!
    assert status != GroundingStatus.GROUNDED
    assert status == GroundingStatus.VALIDATION_FAILED
    assert any("Number '999' absent from cited passage" in (c.unsupported_reason or "") for c in claims)


def test_valid_grounded_answer_is_accepted():
    ev = make_evidence_chunk(
        doc_id="lab04",
        chunk_id="chunk-abc1234",
        text="Під час виконання Лабораторної роботи №4 було створено 17 унікальних генерацій.",
    )
    answer = "У Лабораторній роботі №4 було створено 17 генерацій [lab04#chunk-abc1234]."
    status, cited, invalid, claims = verify_grounding(answer, [ev])
    assert status == GroundingStatus.GROUNDED
    assert cited == ["[lab04#chunk-abc1234]"]
    assert invalid == []
    assert all(c.is_supported for c in claims)


def test_unsupported_question_abstention_receives_insufficient_evidence():
    ev = make_evidence_chunk(
        doc_id="lab04",
        chunk_id="chunk-abc1234",
        text="Під час виконання Лабораторної роботи №4 було створено 17 унікальних генерацій.",
    )
    answer = "I cannot determine the ambient outdoor humidity from the indexed coursework reports."
    status, cited, invalid, claims = verify_grounding(answer, [ev])
    assert status == GroundingStatus.INSUFFICIENT_EVIDENCE
    assert invalid == []
