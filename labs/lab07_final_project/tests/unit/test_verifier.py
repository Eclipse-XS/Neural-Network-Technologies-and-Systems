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



def test_grouped_citations_are_normalized_and_accepted():
    ev1 = make_evidence_chunk("adr001", "chunk-111", "Redis Stack supports HNSW and FLAT indexing.")
    ev2 = make_evidence_chunk("adr001", "chunk-222", "Namespace isolation uses lab07:kb:* prefix.")
    answer = "Redis Stack supports HNSW and namespace isolation [adr001#chunk-111, adr001#chunk-222]."
    status, cited, invalid, claims = verify_grounding(answer, [ev1, ev2])
    assert status == GroundingStatus.GROUNDED
    assert cited == ["[adr001#chunk-111]", "[adr001#chunk-222]"]
    assert invalid == []


def test_pinecone_premise_substitution_hallucination_rejected():
    """Regression test for final_v3_indep_043: Pinecone assertion backed by Redis chunk must FAIL grounding."""
    ev = make_evidence_chunk(
        doc_id="adr_001_vector_and_storage_architecture",
        chunk_id="chunk-312de896eb918af4",
        text="We adopt Redis Stack as the vector database using prefix-based key deletion (lab07:kb:*).",
    )
    answer = (
        "The **Pinecone Serverless vector collection** in ADR 001 is configured for namespace partitioning "
        "using prefix-based key deletion [adr_001_vector_and_storage_architecture#chunk-312de896eb918af4]."
    )
    status, cited, invalid, claims = verify_grounding(answer, [ev])
    assert status != GroundingStatus.GROUNDED
    assert status == GroundingStatus.VALIDATION_FAILED
    assert any("Subject 'pinecone' absent from cited passage" in (c.unsupported_reason or "") for c in claims)


def test_markdown_and_adverb_refusal_classification():
    ev = make_evidence_chunk("lab05", "chunk-1", "Qwen1.5-0.5B-Chat was tested.")
    ans_41 = "The indexed documents do **not** explicitly mention an AWS S3 bucket name."
    ans_45 = "The indexed documents for **Lab 5** do not explicitly report benchmark scores for Gemini 1.5 Pro."
    ans_47 = "The **Milvus distributed cluster** is **not explicitly configured with a consensus protocol**."
    ans_48 = "Інформація про вартість GPT-4o не міститься у відомих документах."
    ans_50 = "В звіті **Lab 3** для моделі Llama 3.1 70B не було прямо зазначено точності."

    for ans in (ans_41, ans_45, ans_47, ans_48, ans_50):
        status, cited, invalid, claims = verify_grounding(ans, [ev])
        assert is_abstention(ans), f"Failed abstention detection on: {ans}"
        assert status == GroundingStatus.INSUFFICIENT_EVIDENCE
        assert invalid == []


def test_url_port_numbers_not_penalized_as_uncited_empirical_claims():
    ev = make_evidence_chunk(
        "lab06_agents_report",
        "chunk-ee8f55a223e64af3",
        "Agent попросив add_numbers(17,25), Python повернув 42.",
    )
    answer = (
        "The agent runs via http://localhost:1234/v1 and tracks tool calls [lab06_agents_report#chunk-ee8f55a223e64af3]."
    )
    status, cited, invalid, claims = verify_grounding(answer, [ev])
    assert status == GroundingStatus.GROUNDED


def test_pinecone_with_trailing_standalone_citation_line_rejected():
    """Regression test: newline before standalone [doc#chunk] must NOT bypass Pinecone check."""
    ev = make_evidence_chunk(
        doc_id="adr_001_vector_and_storage_architecture",
        chunk_id="chunk-312de896eb918af4",
        text="We adopt Redis Stack as the vector database using prefix-based key deletion (lab07:kb:*).",
    )
    answer = (
        "The **Pinecone Serverless vector collection** in ADR 001 is configured with namespace partitioning:\n"
        "- **Prefix deletion:** `lab07:kb:*`\n\n"
        "**[adr_001_vector_and_storage_architecture#chunk-312de896eb918af4]**"
    )
    status, cited, invalid, claims = verify_grounding(answer, [ev])
    assert status != GroundingStatus.GROUNDED
    assert status in (GroundingStatus.VALIDATION_FAILED, GroundingStatus.PARTIALLY_SUPPORTED)
    assert any("Subject 'pinecone' absent" in (c.unsupported_reason or "") for c in claims)


def test_aws_s3_refusal_without_speculative_code_is_insufficient_evidence():
    ev = make_evidence_chunk("api_specification_and_contracts", "chunk-1", "POST /documents/upload accepts .md, .txt, .pdf.")
    ans = "The indexed documents do **not** explicitly mention an AWS S3 bucket name or IAM credentials."
    status, cited, invalid, claims = verify_grounding(ans, [ev])
    assert status == GroundingStatus.INSUFFICIENT_EVIDENCE
    assert invalid == []


def test_absent_enterprise_technologies_caught_against_unrelated_chunks():
    """Verify out-of-corpus technologies (e.g., weaviate, rabbitmq, vllm) are flagged when attributed to corpus chunks."""
    ev_redis = make_evidence_chunk("adr_001", "chunk-r", "We use Redis Stack and FastAPI endpoints.")
    for tech_claim in (
        "Weaviate multi-tenancy shards use Redis Stack [adr_001#chunk-r].",
        "RabbitMQ handles the upload queue on FastAPI [adr_001#chunk-r].",
        "The model is served with vLLM on RTX 3050 [adr_001#chunk-r].",
    ):
        status, _, _, claims = verify_grounding(tech_claim, [ev_redis])
        assert status == GroundingStatus.VALIDATION_FAILED
        assert any("Subject '" in (c.unsupported_reason or "") for c in claims)
