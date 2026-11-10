"""Real LM Studio Acceptance Suite.

Executes all 6 required live acceptance scenarios on the host environment:
1. Full API chat with retrieval and grounded citations.
2. Numerical question with actual calculator tool execution.
3. Cross-document synthesis.
4. Unsupported-query refusal.
5. Context-dependent multi-turn follow-up with SQLite session persistence and isolation.
6. Document ingestion, retrieval, persistence after connection restart, and complete deletion.

All outputs, tool traces, latencies, model identifiers, and failure reasons
are recorded into an auditable evidence JSON file.
"""
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labs.lab07_final_project.src.knowledge_copilot.domain.enums import (
    DocumentStatus,
    GroundingStatus,
    RetrievalMode,
)
from labs.lab07_final_project.src.knowledge_copilot.domain.models import EvidenceChunk
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter
from labs.lab07_final_project.src.knowledge_copilot.application.ingestion_service import IngestionService
from labs.lab07_final_project.src.knowledge_copilot.application.query_service import QueryService
from labs.lab07_final_project.src.knowledge_copilot.agents.orchestrator import AgentOrchestrator
from labs.lab07_final_project.src.knowledge_copilot.rag.embeddings import CopilotEmbeddings
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.llm_client import (
    LocalLLMClient,
    probe_lm_studio_readiness,
    LMStudioStatus,
)
from labs.lab07_final_project.evaluation.eval_generation import (
    audit_citations,
    compute_factual_correctness,
    verify_claim_evidence_support,
    is_refusal_answer,
    check_numerical_match,
    evaluate_generation_gate,
)
from labs.lab07_final_project.evaluation.run_benchmark import ensure_corpus_ingested


async def run_live_acceptance(
    llm_url: str = "http://localhost:1234/v1",
    model_id: str = "ministral-3b",
    redis_url: str = "redis://localhost:6380",
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    output_path = output_path or Path("labs/lab07_final_project/evaluation/baselines/live_acceptance_evidence.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("      LAB 7 REAL LM STUDIO LIVE ACCEPTANCE HARNESS")
    print("==================================================================")

    # 0. Probe LM Studio readiness
    print(f"\n[Step 0] Probing LM Studio at {llm_url} for model '{model_id}'...")
    probe = probe_lm_studio_readiness(llm_url, required_model=model_id)
    print(f"  Probe status: {probe.status.value}")
    print(f"  Probe details: {probe.detail}")
    print(f"  Active model: {probe.model_id}")

    if not probe.is_ready:
        print(f"\n[BLOCKED] LM Studio capability probe failed: {probe.status.value}")
        evidence = {
            "status": "BLOCKED",
            "gate_status": f"BLOCKED ({probe.status.value})",
            "reason": probe.detail,
            "probe_result": {
                "status": probe.status.value,
                "is_ready": probe.is_ready,
                "model_id": probe.model_id,
                "detail": probe.detail,
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scenarios": {},
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(evidence, f, indent=2, ensure_ascii=False)
        return evidence

    active_model = probe.model_id or model_id
    print(f"\n[Step 1] Initializing production components with model '{active_model}'...")
    db_path = Path("labs/lab07_final_project/outputs/benchmark.db")
    repo = SQLiteRepository(db_path)
    embeddings = CopilotEmbeddings(local_files_only=True)
    redis_adapter = RedisVectorAdapter(
        redis_url=redis_url,
        index_name="lab07:benchmark:idx",
        key_prefix="lab07:bench:chunk",
    )
    if not redis_adapter.ping():
        raise RuntimeError(f"Redis is not reachable at {redis_url}")

    redis_adapter.initialize_store(embeddings)
    ingestion_service = IngestionService(
        sqlite_repo=repo,
        redis_adapter=redis_adapter,
        embeddings_model=embeddings,
        require_vector_store=False,
    )

    corpus_path = Path("labs/lab07_final_project/assets/knowledge")
    if len(repo.get_all_chunks()) < 140:
        print("  Ingesting full knowledge corpus into SQLite and Redis...")
        ensure_corpus_ingested(ingestion_service, corpus_path)

    query_service = QueryService(
        ingestion_service=ingestion_service,
        redis_adapter=redis_adapter,
        sqlite_repo=repo,
        default_retrieval_mode=RetrievalMode.HYBRID,
    )

    llm_client = LocalLLMClient(base_url=llm_url, model_id=active_model)
    orchestrator = AgentOrchestrator(
        search_fn=query_service.search,
        list_sources_fn=query_service.list_sources,
        llm_client=llm_client,
        db_path=db_path,
    )
    query_service.orchestrator = orchestrator

    scenarios_results: Dict[str, Any] = {}

    # -------------------------------------------------------------
    # Scenario 1: Full API chat with retrieval and grounded citations
    # -------------------------------------------------------------
    print("\n--- Scenario 1: Full API chat with retrieval and grounded citations ---")
    s1_session = f"live_s1_chat_{int(time.time())}"
    s1_query = "What vector database was chosen in ADR 001?"
    t0 = time.perf_counter()
    turn_1 = await query_service.execute_query(s1_query, session_id=s1_session)
    s1_latency = (time.perf_counter() - t0) * 1000.0

    s1_audit = audit_citations(turn_1.citations, turn_1.evidence)
    s1_fact = compute_factual_correctness(turn_1.canonical_answer, "Redis Stack and SQLite with WAL", [c.chunk.text for c in turn_1.evidence])
    s1_claim = verify_claim_evidence_support(turn_1.canonical_answer, turn_1.evidence)
    s1_tools = [r.tool_name for r in turn_1.tool_executions]

    s1_pass = (
        turn_1.grounding_status != GroundingStatus.EXECUTION_FAILED
        and s1_audit["is_valid"]
        and len(s1_audit["fabricated_citations"]) == 0
        and ("search_knowledge" in s1_tools)
        and ("redis" in turn_1.canonical_answer.casefold())
    )

    scenarios_results["scenario_1_chat_and_citations"] = {
        "query": s1_query,
        "answer": turn_1.canonical_answer,
        "grounding_status": turn_1.grounding_status.value,
        "tools_called": s1_tools,
        "citations": turn_1.citations,
        "audit": s1_audit,
        "factual_correctness": s1_fact,
        "claim_support": s1_claim,
        "latency_ms": round(s1_latency, 2),
        "passed": s1_pass,
    }
    print(f"  Answer: {turn_1.canonical_answer[:120]}...")
    print(f"  Citations: {turn_1.citations}")
    print(f"  Tools: {s1_tools}")
    print(f"  Result: {'PASS' if s1_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # Scenario 2: Numerical question with actual calculator tool execution
    # -------------------------------------------------------------
    print("\n--- Scenario 2: Numerical question with actual calculator tool execution ---")
    s2_session = f"live_s2_calc_{int(time.time())}"
    s2_query = "In the architecture constraints, if the maximum token limit is 8192 and the system instructions use 900 tokens, calculate the exact difference for remaining capacity."
    t0 = time.perf_counter()
    turn_2 = await query_service.execute_query(s2_query, session_id=s2_session)
    s2_latency = (time.perf_counter() - t0) * 1000.0

    s2_tools = [r.tool_name for r in turn_2.tool_executions]
    s2_calc_invoked = "calculate" in s2_tools
    s2_num_match = check_numerical_match(turn_2.canonical_answer, "7292")
    s2_pass = (
        turn_2.grounding_status != GroundingStatus.EXECUTION_FAILED
        and s2_calc_invoked
        and s2_num_match
    )

    scenarios_results["scenario_2_numerical_calculator"] = {
        "query": s2_query,
        "answer": turn_2.canonical_answer,
        "grounding_status": turn_2.grounding_status.value,
        "tools_called": s2_tools,
        "calculator_invoked": s2_calc_invoked,
        "numerical_result_correct": s2_num_match,
        "tool_details": [r.to_dict() for r in turn_2.tool_executions],
        "latency_ms": round(s2_latency, 2),
        "passed": s2_pass,
    }
    print(f"  Answer: {turn_2.canonical_answer}")
    print(f"  Tools: {s2_tools}")
    print(f"  Result: {'PASS' if s2_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # Scenario 3: Cross-document synthesis
    # -------------------------------------------------------------
    print("\n--- Scenario 3: Cross-document synthesis ---")
    s3_session = f"live_s3_cross_{int(time.time())}"
    s3_query = "Compare the models evaluated in Lab 5 versus Lab 6. Which models were tested in each lab?"
    t0 = time.perf_counter()
    turn_3 = await query_service.execute_query(s3_query, session_id=s3_session)
    s3_latency = (time.perf_counter() - t0) * 1000.0

    s3_audit = audit_citations(turn_3.citations, turn_3.evidence)
    s3_tools = [r.tool_name for r in turn_3.tool_executions]
    cited_docs = {Path(c.split("#")[0].lstrip("[")).stem.casefold() for c in turn_3.citations if "#" in c}
    has_lab5 = any("lab05" in d for d in cited_docs) or "lab05" in turn_3.canonical_answer.casefold() or "lab 5" in turn_3.canonical_answer.casefold()
    has_lab6 = any("lab06" in d for d in cited_docs) or "lab06" in turn_3.canonical_answer.casefold() or "lab 6" in turn_3.canonical_answer.casefold()
    s3_pass = (
        turn_3.grounding_status != GroundingStatus.EXECUTION_FAILED
        and has_lab5
        and has_lab6
        and len(s3_audit["fabricated_citations"]) == 0
    )

    scenarios_results["scenario_3_cross_document_synthesis"] = {
        "query": s3_query,
        "answer": turn_3.canonical_answer,
        "grounding_status": turn_3.grounding_status.value,
        "tools_called": s3_tools,
        "citations": turn_3.citations,
        "cited_documents": list(cited_docs),
        "has_lab5_reference": has_lab5,
        "has_lab6_reference": has_lab6,
        "audit": s3_audit,
        "latency_ms": round(s3_latency, 2),
        "passed": s3_pass,
    }
    print(f"  Answer: {turn_3.canonical_answer[:120]}...")
    print(f"  Cited docs: {cited_docs}")
    print(f"  Result: {'PASS' if s3_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # Scenario 4: Unsupported-query refusal
    # -------------------------------------------------------------
    print("\n--- Scenario 4: Unsupported-query refusal ---")
    s4_session = f"live_s4_refusal_{int(time.time())}"
    s4_query = "What were the benchmark results for GPT-4 vision fine-tuning on ImageNet in Lab 6?"
    t0 = time.perf_counter()
    turn_4 = await query_service.execute_query(s4_query, session_id=s4_session)
    s4_latency = (time.perf_counter() - t0) * 1000.0

    s4_refused = is_refusal_answer(turn_4.canonical_answer, turn_4.grounding_status)
    s4_audit = audit_citations(turn_4.citations, turn_4.evidence)
    s4_pass = s4_refused and len(s4_audit["fabricated_citations"]) == 0

    scenarios_results["scenario_4_unsupported_refusal"] = {
        "query": s4_query,
        "answer": turn_4.canonical_answer,
        "grounding_status": turn_4.grounding_status.value,
        "is_refusal": s4_refused,
        "citations": turn_4.citations,
        "fabricated_citations_count": len(s4_audit["fabricated_citations"]),
        "latency_ms": round(s4_latency, 2),
        "passed": s4_pass,
    }
    print(f"  Answer: {turn_4.canonical_answer}")
    print(f"  Refused: {s4_refused}")
    print(f"  Result: {'PASS' if s4_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # Scenario 5: Context-dependent multi-turn follow-up
    # -------------------------------------------------------------
    print("\n--- Scenario 5: Context-dependent multi-turn follow-up ---")
    s5_session = f"live_s5_multiturn_{int(time.time())}"
    t0 = time.perf_counter()

    # Turn 5a: Context anchor
    t5a_query = "What vector database was chosen in ADR 001?"
    turn_5a = await query_service.execute_query(t5a_query, session_id=s5_session)

    # Turn 5b: Contextual follow-up requiring Turn 5a history to resolve "it"
    t5b_query = "Why was it selected instead of Milvus and Qdrant? Check ADR 001."
    turn_5b = await query_service.execute_query(t5b_query, session_id=s5_session)
    s5_latency = (time.perf_counter() - t0) * 1000.0

    # Verify model-visible conversation history persisted in SQLite
    sess_record = repo.get_session(s5_session)
    history_ok = sess_record is not None and len(sess_record.turns) >= 2

    # Verify session isolation: same follow-up in fresh session
    iso_session = f"live_s5_isolated_{int(time.time())}"
    turn_iso = await query_service.execute_query(t5b_query, session_id=iso_session)
    iso_record = repo.get_session(iso_session)
    isolation_ok = iso_record is not None and len(iso_record.turns) == 1

    t5b_audit = audit_citations(turn_5b.citations, turn_5b.evidence)
    t5b_relevant = any(w in turn_5b.canonical_answer.casefold() for w in ["milvus", "qdrant", "memory", "ram", "overhead", "footprint", "redis"])

    s5_pass = (
        history_ok
        and isolation_ok
        and t5b_relevant
        and turn_5b.grounding_status != GroundingStatus.EXECUTION_FAILED
        and len(t5b_audit["fabricated_citations"]) == 0
    )

    scenarios_results["scenario_5_contextual_multiturn"] = {
        "turn_1": {
            "query": t5a_query,
            "answer": turn_5a.canonical_answer,
            "citations": turn_5a.citations,
        },
        "turn_2": {
            "query": t5b_query,
            "answer": turn_5b.canonical_answer,
            "citations": turn_5b.citations,
            "audit": t5b_audit,
        },
        "history_persisted_in_sqlite": history_ok,
        "session_isolation_verified": isolation_ok,
        "context_resolved_correctly": t5b_relevant,
        "latency_ms": round(s5_latency, 2),
        "passed": s5_pass,
    }
    print(f"  Turn 1: {turn_5a.canonical_answer[:100]}...")
    print(f"  Turn 2: {turn_5b.canonical_answer[:100]}...")
    print(f"  History persisted: {history_ok}, Isolation: {isolation_ok}")
    print(f"  Result: {'PASS' if s5_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # Scenario 6: Document ingestion, retrieval, persistence after restart, and deletion
    # -------------------------------------------------------------
    print("\n--- Scenario 6: Document lifecycle, retrieval, persistence, and deletion ---")
    s6_t0 = time.perf_counter()
    doc_id = "doc_live_acceptance_lifecycle"
    doc_content = b"# Live Acceptance Telemetry Protocol\n\nQuantum telemetry confirms benchmark latency delta is 14.8 milliseconds in test sector 9."

    # 6a: Ingest
    ingested_doc = ingestion_service.ingest_file(
        filename="live_telemetry_protocol.md",
        content=doc_content,
        document_id=doc_id,
        collection="test_lifecycle",
    )
    ingest_ok = ingested_doc.status == DocumentStatus.INDEXED and ingested_doc.chunk_count > 0

    # 6b: Retrieve
    retrieved = query_service.search(
        query="quantum telemetry latency delta test sector 9",
        k=3,
        document_id=doc_id,
    )
    retrieval_ok = len(retrieved) >= 1 and "14.8 milliseconds" in retrieved[0].chunk.text

    # 6c: Persistence after restart (instantiate fresh repository and redis adapter)
    fresh_repo = SQLiteRepository(db_path)
    fresh_redis = RedisVectorAdapter(
        redis_url=redis_url,
        index_name="lab07:benchmark:idx",
        key_prefix="lab07:bench:chunk",
    )
    fresh_doc = fresh_repo.get_document(doc_id)
    fresh_chunks = fresh_repo.get_chunks_for_document(doc_id)
    persistence_ok = fresh_doc is not None and len(fresh_chunks) > 0

    # 6d: Deletion and verification of complete purge
    ingestion_service.delete_document(doc_id)
    purged_doc = fresh_repo.get_document(doc_id)
    purged_chunks = fresh_repo.get_chunks_for_document(doc_id)
    deleted_ok = purged_doc is None and len(purged_chunks) == 0

    s6_latency = (time.perf_counter() - s6_t0) * 1000.0
    s6_pass = ingest_ok and retrieval_ok and persistence_ok and deleted_ok

    scenarios_results["scenario_6_document_lifecycle"] = {
        "ingest_successful": ingest_ok,
        "retrieval_successful": retrieval_ok,
        "persistence_after_restart_verified": persistence_ok,
        "complete_purge_verified": deleted_ok,
        "latency_ms": round(s6_latency, 2),
        "passed": s6_pass,
    }
    print(f"  Ingest: {ingest_ok}, Retrieve: {retrieval_ok}, Persistence: {persistence_ok}, Purge: {deleted_ok}")
    print(f"  Result: {'PASS' if s6_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # Overall summary and gate evaluation
    # -------------------------------------------------------------
    all_passed = all(sc.get("passed", False) for sc in scenarios_results.values())
    overall_status = "PASS" if all_passed else "FAIL"

    evidence = {
        "status": "VERIFIED_LIVE",
        "gate_status": overall_status,
        "model_id": active_model,
        "runtime_config": {
            "llm_url": llm_url,
            "redis_url": redis_url,
            "retrieval_mode": "hybrid",
            "db_path": str(db_path),
        },
        "scenarios": scenarios_results,
        "summary": {
            "total_scenarios": len(scenarios_results),
            "passed_scenarios": sum(1 for sc in scenarios_results.values() if sc.get("passed")),
            "failed_scenarios": sum(1 for sc in scenarios_results.values() if not sc.get("passed")),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evidence, f, indent=2, ensure_ascii=False)

    print("\n==================================================================")
    print(f"  OVERALL LM STUDIO ACCEPTANCE GATE STATUS: [{overall_status}]")
    print("==================================================================")
    print(f"Evidence successfully written to: {output_path}")

    return evidence


if __name__ == "__main__":
    asyncio.run(run_live_acceptance())
