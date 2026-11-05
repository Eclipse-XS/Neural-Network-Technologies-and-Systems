"""Comprehensive Evaluation Integrity, Schema, and Monotonicity Tests."""
import json
import math
from pathlib import Path
import pytest

from labs.lab07_final_project.evaluation.metrics import (
    compute_retrieval_metrics,
    is_chunk_relevant,
    normalize_text,
)
from labs.lab07_final_project.src import evaluation


def test_baseline_placeholder_citations_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "LAB_ROOT", tmp_path)
    (tmp_path / "trace.json").write_text("[]", encoding="utf-8")
    for answer in ["A [source_id#chunk_2]", "A [lab4_report#1]", "source_id: *L4_2023*"]:
        result = evaluation.citation_audit(dict(final_answer=answer, trace_path="trace.json"))
        assert result["citation"] == "FAIL"
        assert result["invalid_citations"]


def test_current_trace_membership(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "LAB_ROOT", tmp_path)
    events = [
        dict(
            event_type="tool_result",
            tool_name="search_knowledge",
            result={"chunks": [{"citation": "[lab04#chunk-real]"}]},
        )
    ]
    (tmp_path / "trace.json").write_text(json.dumps(events), encoding="utf-8")
    record = dict(final_answer="Claim [lab04#chunk-real]; other [lab04#chunk-other]", trace_path="trace.json")
    result = evaluation.citation_audit(record)
    assert result["citation"] == "PARTIAL"
    assert result["invalid_citations"] == ["[lab04#chunk-other]"]


# =========================================================================
# P0-1: RECALL@K MONOTONICITY REGRESSION TESTS
# =========================================================================

@pytest.mark.parametrize(
    "rel_ranks_doc1, rel_ranks_doc2",
    [
        ([1], []),            # Single doc: rank 1 -> 1.0, 1.0, 1.0
        ([2], []),            # Single doc: rank 2 -> 0.0, 1.0, 1.0
        ([4], []),            # Single doc: rank 4 -> 0.0, 0.0, 1.0
        ([6], []),            # Single doc: unretrieved -> 0.0, 0.0, 0.0
        ([1], [3]),           # Multi doc: rank 1 & 3 -> 0.5, 1.0, 1.0
        ([1], [5]),           # Multi doc: rank 1 & 5 -> 0.5, 0.5, 1.0
        ([2], [4]),           # Multi doc: rank 2 & 4 -> 0.0, 0.5, 1.0
        ([3], [6]),           # Multi doc: rank 3 & unretrieved -> 0.0, 0.5, 0.5
        ([6], [6]),           # Multi doc: both unretrieved -> 0.0, 0.0, 0.0
    ],
)
def test_recall_monotonicity_single_and_multi_doc(rel_ranks_doc1, rel_ranks_doc2):
    """Enforce Recall@1 <= Recall@3 <= Recall@5 for every single query."""
    is_multi = len(rel_ranks_doc2) > 0
    relevant_docs = ["doc_a", "doc_b"] if is_multi else ["doc_a"]
    
    # Construct synthetic retrieved chunks (ranks 1 to 5)
    retrieved_chunks = []
    for rank in range(1, 6):
        if rank in rel_ranks_doc1:
            retrieved_chunks.append({
                "chunk_id": f"chunk_a_{rank}",
                "document_id": "doc_a",
                "filename": "doc_a.md",
                "content": f"evidence for doc_a at rank {rank}",
                "section": "Sec A",
            })
        elif rank in rel_ranks_doc2:
            retrieved_chunks.append({
                "chunk_id": f"chunk_b_{rank}",
                "document_id": "doc_b",
                "filename": "doc_b.md",
                "content": f"evidence for doc_b at rank {rank}",
                "section": "Sec B",
            })
        else:
            retrieved_chunks.append({
                "chunk_id": f"chunk_other_{rank}",
                "document_id": "other_doc",
                "filename": "other_doc.md",
                "content": f"irrelevant content at rank {rank}",
                "section": "Other Sec",
            })

    gold_query = {
        "query_id": "test_q",
        "category": "cross_document" if is_multi else "exact_fact",
        "relevant_doc_ids": relevant_docs,
        "evidence_spans": ["evidence for doc_a", "evidence for doc_b"] if is_multi else ["evidence for doc_a"],
        "gold_passage_ids": [f"chunk_a_{r}" for r in rel_ranks_doc1] + [f"chunk_b_{r}" for r in rel_ranks_doc2],
        "refusal_expected": False,
        "retrieved_chunks": retrieved_chunks,
    }

    metrics = compute_retrieval_metrics([gold_query], k_thresholds=(1, 3, 5))
    r1 = metrics["evidence_recall_1"]
    r3 = metrics["evidence_recall_3"]
    r5 = metrics["evidence_recall_5"]

    assert r1 <= r3 + 1e-9 <= r5 + 1e-9, f"Monotonicity violation: R@1={r1}, R@3={r3}, R@5={r5}"


# =========================================================================
# P0-2: DOCUMENT-ONLY FALSE POSITIVE REJECTION
# =========================================================================

def test_final_001_status_chunk_rejected_as_irrelevant():
    """Regression test for final_001: ADR Status section ('Accepted') must NOT establish relevance."""
    chunk_1_status = {
        "chunk_id": "chunk-1fc9181f6057d769",
        "document_id": "adr_001_vector_and_storage_architecture",
        "filename": "adr_001_vector_and_storage_architecture.md",
        "section": "ADR 001: Vector Storage and Relational Persistence Architecture / Status",
        "content": "## ADR 001: Vector Storage and Relational Persistence Architecture / Status\n# ADR 001: Vector Storage and Relational Persistence Architecture  \n## Status\nAccepted",
    }
    
    gold_query_final_001 = {
        "id": "final_001",
        "relevant_doc_ids": ["adr_001_vector_and_storage_architecture.md"],
        "target_sections": ["## Decision"],
        "evidence_spans": [
            "**Redis Stack (Search & JSON)** as the vector database:",
            "**SQLite with WAL (Write-Ahead Logging)** for application state and chat memory:",
        ],
        "gold_passage_ids": ["chunk-312de896eb918af4"],
        "refusal_expected": False,
    }

    # Status chunk must be rejected
    assert is_chunk_relevant(chunk_1_status, gold_query_final_001) is False, (
        "Document name or status section title alone must NEVER establish factual evidence relevance!"
    )

    # Actual decision chunk must be accepted
    chunk_decision = {
        "chunk_id": "chunk-312de896eb918af4",
        "document_id": "adr_001_vector_and_storage_architecture",
        "filename": "adr_001_vector_and_storage_architecture.md",
        "section": "ADR 001: Vector Storage and Relational Persistence Architecture / Decision",
        "content": "1. **Redis Stack (Search & JSON)** as the vector database:\n2. **SQLite with WAL (Write-Ahead Logging)** for application state and chat memory:",
    }
    assert is_chunk_relevant(chunk_decision, gold_query_final_001) is True


def test_document_match_alone_returns_false():
    """Ensure a chunk belonging to a relevant document without gold passages or evidence spans returns False."""
    chunk = {
        "chunk_id": "random_chunk_id",
        "document_id": "lab03_rag_report",
        "filename": "lab03_rag_report.md",
        "section": "Random Section",
        "content": "This text has nothing to do with the question or gold targets.",
    }
    query = {
        "id": "test_q",
        "relevant_doc_ids": ["lab03_rag_report.md"],
        "evidence_spans": ["Specific metric: Recall@5 is 0.85"],
        "gold_passage_ids": ["chunk_specific_123"],
        "refusal_expected": False,
    }
    assert is_chunk_relevant(chunk, query) is False


# =========================================================================
# P0-3 & P0-5: DATASET INTEGRITY AND PRODUCTION CHUNK VALIDATION
# =========================================================================

def test_evaluation_dataset_annotations_and_hashes_valid():
    """Verify that all evaluation datasets pass read-only validation against the manifest and corpus chunks."""
    from labs.lab07_final_project.evaluation.validate_annotations import (
        DATASET_FILES,
        MANIFEST_PATH,
        load_source_documents,
        generate_production_chunks,
        validate_dataset_file,
    )
    from labs.lab07_final_project.src.config import Settings
    from labs.lab07_final_project.src.knowledge_copilot.rag.embeddings import CopilotEmbeddings

    assert MANIFEST_PATH.exists(), "dataset_integrity_manifest.json must exist"
    trusted_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    settings = Settings()
    embeddings = CopilotEmbeddings(
        model_name=settings.embedding_model,
        revision=settings.embedding_revision,
        device="cpu",
        local_files_only=True,
    )
    docs = load_source_documents()
    chunks_by_id, chunks_by_doc, *_ = generate_production_chunks(
        settings, embeddings, allow_cached_manifest=True
    )

    for fname in DATASET_FILES:
        res = validate_dataset_file(
            filename=fname,
            chunks_by_id=chunks_by_id,
            chunks_by_doc=chunks_by_doc,
            docs=docs,
            trusted_manifest=trusted_manifest,
        )
        assert res["status"] == "VALIDATED"


# =========================================================================
# P0-5: BENCHMARK RESULT SCHEMA AND SUMMARY CONSISTENCY
# =========================================================================

def test_benchmark_result_schema_and_recomputation_consistency():
    """Verify that committed benchmark results match raw itemized recomputations exactly."""
    benchmark_dir = Path("labs/lab07_final_project/evaluation/baselines")
    benchmark_files = list(benchmark_dir.glob("*benchmark_results.json"))
    
    assert len(benchmark_files) >= 1, "At least one benchmark result artifact must exist."
    
    for b_file in benchmark_files:
        data = json.loads(b_file.read_text(encoding="utf-8"))
        assert "summary" in data or "evidence_recall_1" in data, f"{b_file.name} missing summary fields"
        assert "itemized_results" in data, f"{b_file.name} missing raw itemized_results"

        items = data["itemized_results"]
        assert len(items) > 0, f"{b_file.name} itemized_results is empty"

        # Recompute metrics directly from itemized results
        recomputed = compute_retrieval_metrics(items, k_thresholds=(1, 3, 5))
        summary = data.get("summary", data)

        assert math.isclose(summary["evidence_recall_1"], recomputed["evidence_recall_1"], rel_tol=1e-3), (
            f"{b_file.name}: Summary Recall@1 {summary['evidence_recall_1']} does not match raw itemized {recomputed['evidence_recall_1']}"
        )
        assert math.isclose(summary["evidence_recall_5"], recomputed["evidence_recall_5"], rel_tol=1e-3), (
            f"{b_file.name}: Summary Recall@5 {summary['evidence_recall_5']} does not match raw itemized {recomputed['evidence_recall_5']}"
        )
        assert math.isclose(summary["mrr"], recomputed["mrr"], rel_tol=1e-3), (
            f"{b_file.name}: Summary MRR {summary['mrr']} does not match raw itemized {recomputed['mrr']}"
        )


# =========================================================================
# P0-1: GENERATION EVALUATION ACCEPTANCE GATE THRESHOLD REGRESSIONS
# =========================================================================

from labs.lab07_final_project.evaluation.eval_generation import (
    evaluate_generation_gate,
    compute_factual_correctness,
    is_refusal_answer,
    verify_claim_evidence_support,
    GENERATION_ACCEPTANCE_THRESHOLDS,
)
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, EvidenceChunk, Score
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.llm_client import (
    probe_lm_studio_readiness,
    LMStudioStatus,
)


def _build_mock_summary(
    exec_failures=0,
    fact_correct=0.85,
    claim_supp=0.90,
    citation_val=0.90,
    refusal_rate=0.95,
    calc_rate=0.85,
):
    return {
        "execution_failures": exec_failures,
        "metrics": {
            "factual_answer_correctness": {"value": fact_correct},
            "claim_support_rate": {"value": claim_supp},
            "citation_validity_rate": {"value": citation_val},
            "unsupported_refusal_rate": {"value": refusal_rate},
            "calculator_usage_rate": {"value": calc_rate},
        },
    }


def test_generation_gate_pass_when_all_thresholds_satisfied():
    summary = _build_mock_summary()
    status, violations = evaluate_generation_gate(summary)
    assert status == "PASS"
    assert len(violations) == 0


def test_generation_gate_fails_on_execution_failure():
    summary = _build_mock_summary(exec_failures=1)
    status, violations = evaluate_generation_gate(summary)
    assert status == "FAIL"
    assert any("Execution failures: 1" in v for v in violations)


@pytest.mark.parametrize(
    "metric_arg, metric_name, bad_value",
    [
        ("fact_correct", "factual_answer_correctness", 0.79),
        ("claim_supp", "claim_support_rate", 0.84),
        ("citation_val", "citation_validity_rate", 0.84),
        ("refusal_rate", "unsupported_refusal_rate", 0.89),
        ("calc_rate", "calculator_usage_rate", 0.79),
    ],
)
def test_generation_gate_fails_on_subthreshold_metric(metric_arg, metric_name, bad_value):
    kwargs = {metric_arg: bad_value}
    summary = _build_mock_summary(**kwargs)
    status, violations = evaluate_generation_gate(summary)
    assert status == "FAIL"
    assert any(metric_name in v for v in violations)


def test_generation_gate_fails_when_metric_is_none():
    summary = _build_mock_summary()
    summary["metrics"]["factual_answer_correctness"]["value"] = None
    status, violations = evaluate_generation_gate(summary)
    assert status == "FAIL"
    assert any("factual_answer_correctness: None" in v for v in violations)


def test_factual_correctness_and_refusal_evaluations():
    assert is_refusal_answer("Інформація не знайдена в індексованій документації.", GroundingStatus.INSUFFICIENT_EVIDENCE)
    assert is_refusal_answer("This is not found in the technical documents.", GroundingStatus.GROUNDED)
    assert not is_refusal_answer("Redis Stack was selected for vector search.", GroundingStatus.GROUNDED)

    assert compute_factual_correctness(
        answer="Redis Stack (Search & JSON) was chosen with SQLite WAL.",
        expected="Redis Stack was chosen for vector storage and SQLite with WAL for application state.",
        evidence_spans=["Redis Stack (Search & JSON) as the vector database"],
    )

    # Missing critical numerical fact
    assert not compute_factual_correctness(
        answer="Latency was 99.9 milliseconds.",
        expected="Latency was 14.2 milliseconds.",
        evidence_spans=["14.2 milliseconds"],
    )


def test_granular_lm_studio_probe_statuses(monkeypatch):
    import httpx

    # 1. Unreachable server
    def mock_get_fail(*args, **kwargs):
        raise httpx.ConnectError("Connection refused")
    monkeypatch.setattr(httpx, "get", mock_get_fail)
    res = probe_lm_studio_readiness("http://localhost:1234/v1")
    assert res.status == LMStudioStatus.SERVER_UNREACHABLE
    assert not res.is_ready

    # 2. Server up, but no models loaded
    class MockResp:
        status_code = 200
        def json(self):
            return {"data": []}
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: MockResp())
    res_no_model = probe_lm_studio_readiness("http://localhost:1234/v1")
    assert res_no_model.status == LMStudioStatus.NO_MODEL_LOADED
    assert not res_no_model.is_ready

    # 3. Model not found
    class MockRespModels:
        status_code = 200
        def json(self):
            return {"data": [{"id": "other-model-7b"}]}
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: MockRespModels())
    res_not_found = probe_lm_studio_readiness("http://localhost:1234/v1", required_model="ministral-3b")
    assert res_not_found.status == LMStudioStatus.MODEL_NOT_FOUND
    assert not res_not_found.is_ready

