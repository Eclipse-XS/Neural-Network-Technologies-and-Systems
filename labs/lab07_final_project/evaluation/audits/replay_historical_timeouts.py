"""Replay historical cross-document timeout queries against live system.

Demonstrates that queries final_021, final_022, final_025, final_027, and final_030,
which historically aborted on the 120.0s agent loop timeout, now complete reliably
under the 180.0s resilient timeout budget with full cross-document citations and grounding.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus, RetrievalMode
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter
from labs.lab07_final_project.src.knowledge_copilot.application.ingestion_service import IngestionService
from labs.lab07_final_project.src.knowledge_copilot.application.query_service import QueryService
from labs.lab07_final_project.src.knowledge_copilot.agents.orchestrator import AgentOrchestrator
from labs.lab07_final_project.src.knowledge_copilot.rag.embeddings import CopilotEmbeddings
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.llm_client import LocalLLMClient
from labs.lab07_final_project.evaluation.eval_generation import audit_citations, compute_factual_correctness


TARGET_IDS = ["final_021", "final_022", "final_025", "final_027", "final_030"]


async def main():
    dataset_path = Path("labs/lab07_final_project/evaluation/datasets/final_acceptance_holdout_50.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_queries = json.load(f)

    target_queries = [q for q in all_queries if q["id"] in TARGET_IDS]
    print(f"Loaded {len(target_queries)} historical timeout queries for replay.")

    # Initialize live services
    db_path = Path("labs/lab07_final_project/outputs/benchmark.db")
    repo = SQLiteRepository(db_path)
    embeddings = CopilotEmbeddings(local_files_only=True)
    redis_adapter = RedisVectorAdapter(
        redis_url="redis://127.0.0.1:6379",
        index_name="lab07:benchmark:idx",
        key_prefix="lab07:bench:chunk",
    )
    redis_adapter.initialize_store(embeddings)

    ingestion_service = IngestionService(
        sqlite_repo=repo,
        redis_adapter=redis_adapter,
        embeddings_model=embeddings,
        require_vector_store=False,
    )

    query_service = QueryService(
        ingestion_service=ingestion_service,
        redis_adapter=redis_adapter,
        sqlite_repo=repo,
        default_retrieval_mode=RetrievalMode.HYBRID,
    )

    llm_client = LocalLLMClient(base_url="http://127.0.0.1:1234/v1", model_id="mistralai/ministral-3-3b")
    orchestrator = AgentOrchestrator(
        search_fn=query_service.search,
        list_sources_fn=query_service.list_sources,
        llm_client=llm_client,
        db_path=db_path,
    )
    query_service.orchestrator = orchestrator

    replay_results = []
    print("\n" + "=" * 70)
    print("STARTING HISTORICAL TIMEOUT REPLAY")
    print("=" * 70)

    for item in target_queries:
        qid = item["id"]
        q_text = item["query"]
        expected = item.get("expected_answer", "")
        evidence_spans = item.get("evidence_spans", [])
        rel_docs = [Path(d).stem.casefold() for d in item.get("relevant_doc_ids", [])]

        print(f"\n--- Replaying {qid} ---")
        print(f"Query: {q_text}")
        session_id = f"replay_{qid}_{int(time.time())}"

        t0 = time.perf_counter()
        turn = await query_service.execute_query(
            query=q_text,
            session_id=session_id,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        total_time_ms = (time.perf_counter() - t0) * 1000.0

        audit = audit_citations(turn.citations, turn.evidence)
        valid_cited_docs = set()
        for c_tag in audit["valid_citations"]:
            doc_id = c_tag.split("#")[0].strip("[]").casefold()
            valid_cited_docs.add(doc_id)
        doc_coverage = (
            sum(1 for d in rel_docs if d in valid_cited_docs) / len(rel_docs)
            if rel_docs else 1.0
        )

        fact_ok = compute_factual_correctness(
            answer=turn.canonical_answer,
            expected=expected,
            evidence_spans=evidence_spans,
        )

        record = {
            "query_id": qid,
            "query": q_text,
            "historical_status": "TIMEOUT_ABORT (>120s)",
            "replay_status": turn.grounding_status.value,
            "total_latency_ms": round(total_time_ms, 2),
            "total_latency_seconds": round(total_time_ms / 1000.0, 2),
            "within_180s_budget": total_time_ms <= 180000.0,
            "tools_called": [rec.tool_name for rec in turn.tool_executions],
            "citations_found": turn.citations,
            "valid_citations": audit["valid_citations"],
            "fabricated_citations": audit["fabricated_citations"],
            "citation_validity_rate": audit["validity_rate"],
            "cross_doc_coverage": round(doc_coverage, 4),
            "factual_correctness": fact_ok,
            "answer_preview": turn.canonical_answer[:300] + "..." if len(turn.canonical_answer) > 300 else turn.canonical_answer,
            "full_answer": turn.canonical_answer,
        }
        replay_results.append(record)

        print(f"Status: {turn.grounding_status.value}")
        print(f"Latency: {total_time_ms/1000.0:.2f}s (budget: 180.0s)")
        print(f"Citations: {len(turn.citations)} (valid: {len(audit['valid_citations'])})")
        print(f"Cross-doc coverage: {doc_coverage:.2f}")
        print(f"Factual correctness: {fact_ok}")

    output_dir = Path("labs/lab07_final_project/evaluation/audits")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "timeout_recovery_replay_traces.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "model_id": "mistralai/ministral-3-3b",
                "total_replayed": len(replay_results),
                "successful_replays": sum(1 for r in replay_results if r["replay_status"] != "EXECUTION_FAILED"),
                "all_within_budget": all(r["within_180s_budget"] for r in replay_results),
                "traces": replay_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nSaved replay traces to {out_file}")


if __name__ == "__main__":
    asyncio.run(main())
