"""Real Answer-Level Generation Evaluation Suite.

Executes genuine local agent pipeline:
FastAPI / QueryService -> AgentOrchestrator -> Runner.run -> LM Studio

Grounding and Evidence Rules:
1. No answer-producing code may access ground truth labels (refusal_expected, expected_answer, relevant_doc_ids).
2. All tool executions, citations, and answers must be extracted from actual runtime execution traces.
3. If LM Studio is unavailable on the configured port, marks all generation metrics as NOT_VERIFIED
   and the live gate as BLOCKED (<granular_status>). No synthetic templates or fabricated outputs.
4. Gate status is strictly threshold-based. PASS requires:
   - 0 execution failures
   - factual_answer_correctness >= 0.80
   - claim_support_rate >= 0.85
   - citation_validity_rate >= 0.85
   - unsupported_refusal_rate >= 0.90
   - calculator_usage_rate >= 0.80 (for numerical queries)
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus, RetrievalMode
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Document, Chunk, EvidenceChunk
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
    LMStudioProbeResult,
)
from labs.lab07_final_project.evaluation.metrics import normalize_text

GENERATION_ACCEPTANCE_THRESHOLDS = {
    "execution_failures": 0,
    "factual_answer_correctness": 0.80,
    "claim_support_rate": 0.85,
    "citation_validity_rate": 0.85,
    "unsupported_refusal_rate": 0.90,
    "calculator_usage_rate": 0.80,
}


def is_refusal_answer(answer: str, grounding_status: GroundingStatus) -> bool:
    """Evaluate whether an answer appropriately abstained on an unsupported query."""
    if grounding_status == GroundingStatus.INSUFFICIENT_EVIDENCE:
        return True
    refusal_patterns = [
        r"не знайдено",
        r"не містить",
        r"відсутн\w* (?:інформаці|дані)",
        r"insufficient evidence",
        r"not found in (?:the )?(?:indexed |technical )?documents",
        r"no evidence",
        r"abstain",
        r"cannot answer",
        r"не можу надати",
        r"немає інформації",
    ]
    a_lower = (answer or "").casefold()
    return any(re.search(pat, a_lower) for pat in refusal_patterns)


def compute_factual_correctness(answer: str, expected: str, evidence_spans: List[str]) -> bool:
    """Evaluate whether the answer factual content aligns with gold expected answer and evidence spans."""
    a_norm = normalize_text(answer)
    e_norm = normalize_text(expected)
    if not a_norm or not e_norm:
        return False

    # Check for direct key numbers or acronyms
    numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", e_norm))
    if numbers:
        matched_numbers = sum(1 for n in numbers if n in a_norm)
        if matched_numbers < len(numbers):
            return False

    # 1. Direct span overlap
    if evidence_spans:
        for span in evidence_spans:
            norm_span = normalize_text(span)
            if norm_span and norm_span in a_norm:
                return True

    # 2. Key tokens overlap
    tokens_expected = set(re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", e_norm))
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "has", "have", "had", "not", "but", "what", "which", "who", "when", "where",
        "було", "для", "або", "при", "який", "яка", "яке", "які", "так", "ні", "був",
    }
    content_tokens = {t for t in tokens_expected if t not in stopwords}
    if not content_tokens:
        return e_norm in a_norm

    matched = sum(1 for t in content_tokens if t in a_norm)
    token_overlap = matched / len(content_tokens)
    return token_overlap >= 0.50


def verify_claim_evidence_support(answer: str, evidence: List[EvidenceChunk]) -> bool:
    """Verify that substantive claims in answer are anchored in retrieved chunks."""
    if not evidence:
        return False
    combined_ev = normalize_text(" ".join(c.chunk.text for c in evidence))
    a_norm = normalize_text(answer)

    # Extract key technical phrases from answer
    key_phrases = re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{4,}\b", a_norm)
    if not key_phrases:
        return True

    matched = sum(1 for p in key_phrases if p in combined_ev)
    return (matched / len(key_phrases)) >= 0.50


def check_numerical_match(answer: str, expected: str) -> bool:
    """Verify that expected numerical results appear in the answer."""
    exp_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", expected)
    ans_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", answer)
    if not exp_numbers:
        return True
    return all(n in ans_numbers for n in exp_numbers)


def evaluate_generation_gate(metrics_summary: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Determine PASS/FAIL status based on explicit acceptance thresholds.
    
    Returns (gate_status, list_of_violations).
    """
    violations = []
    
    # 1. Execution failures must be 0
    exec_failures = metrics_summary.get("execution_failures", 0)
    if exec_failures > 0:
        violations.append(f"Execution failures: {exec_failures} (target: 0)")

    metrics = metrics_summary.get("metrics", {})

    for metric_name, threshold in GENERATION_ACCEPTANCE_THRESHOLDS.items():
        if metric_name == "execution_failures":
            continue
        metric_data = metrics.get(metric_name, {})
        val = metric_data.get("value")
        if val is None:
            violations.append(f"{metric_name}: None (target >= {threshold:.2f})")
        elif val < threshold:
            violations.append(f"{metric_name}: {val:.4f} (target >= {threshold:.2f})")

    if not violations:
        return "PASS", []
    return "FAIL", violations


async def run_generation_evaluation(
    dataset_path: Path,
    output_path: Path,
    llm_url: str = "http://localhost:1234/v1",
    model_id: Optional[str] = "ministral-3b",
) -> Dict[str, Any]:
    dataset = json.load(open(dataset_path, encoding="utf-8"))
    print(f"Loaded {len(dataset)} queries from {dataset_path.name}")

    probe = probe_lm_studio_readiness(base_url=llm_url, required_model=model_id)

    if not probe.is_ready:
        print("\n" + "=" * 70)
        print(f" [NOTICE] Local LLM is NOT READY: {probe.status.value}")
        print(f" Detail: {probe.detail}")
        print(" Evidence contract prohibits substituting synthetic templates or fake answers.")
        print(" Marking real generation metrics as NOT_VERIFIED.")
        print(f" Marking live acceptance gate as BLOCKED ({probe.status.value}).")
        print("=" * 70 + "\n")

        offline_result = {
            "status": "NOT_VERIFIED",
            "gate_status": f"BLOCKED ({probe.status.value})",
            "probe_status": probe.status.value,
            "reason": probe.detail,
            "model_id": probe.model_id,
            "available_models": probe.available_models,
            "execution_failures": 0,
            "metrics": {
                "factual_answer_correctness": {"numerator": 0, "denominator": 0, "value": None, "status": "NOT_VERIFIED"},
                "claim_support_rate": {"numerator": 0, "denominator": 0, "value": None, "status": "NOT_VERIFIED"},
                "citation_validity_rate": {"numerator": 0, "denominator": 0, "value": None, "status": "NOT_VERIFIED"},
                "unsupported_refusal_rate": {"numerator": 0, "denominator": 0, "value": None, "status": "NOT_VERIFIED"},
                "calculator_usage_rate": {"numerator": 0, "denominator": 0, "value": None, "status": "NOT_VERIFIED"},
                "cross_document_coverage": {"numerator": 0, "denominator": 0, "value": None, "status": "NOT_VERIFIED"},
                "latency": {"p50_ms": None, "p95_ms": None, "mean_ms": None, "status": "NOT_VERIFIED"},
            },
            "evaluated_queries_count": len(dataset),
            "executed_live_turns": 0,
            "itemized_results": [],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(offline_result, f, indent=2, ensure_ascii=False)

        return offline_result

    # LM Studio is verified online and ready
    active_model = probe.model_id or model_id or "ministral-3b"
    print(f"\nLocal LLM verified ready with active model: {active_model}. Initializing production pipeline...")
    db_path = Path("labs/lab07_final_project/outputs/benchmark.db")
    repo = SQLiteRepository(db_path)
    embeddings = CopilotEmbeddings(local_files_only=True)
    redis_adapter = RedisVectorAdapter(
        redis_url="redis://localhost:6380",
        index_name="lab07:benchmark:idx",
        key_prefix="lab07:benchmark:chunk",
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

    llm_client = LocalLLMClient(base_url=llm_url, model_id=active_model)
    orchestrator = AgentOrchestrator(
        search_fn=query_service.search,
        list_sources_fn=query_service.list_sources,
        llm_client=llm_client,
        db_path=db_path,
    )
    query_service.orchestrator = orchestrator

    itemized_results = []
    latencies = []
    refusal_evals = []
    citation_evals = []
    factual_evals = []
    claim_evals = []
    calculator_evals = []
    cross_doc_evals = []
    execution_failures = 0

    print(f"Executing real agent turns across {len(dataset)} queries...")

    for idx, item in enumerate(dataset, 1):
        q_text = item["query"]
        qid = item.get("id", f"eval_{idx:03d}")
        session_id = f"eval_{qid}_{int(time.time())}"
        is_refusal_req = item.get("refusal_expected", False)
        is_calc_req = item.get("category") == "numerical_reasoning" or "calculate" in item.get("expected_tool_calls", [])
        is_cross_doc = item.get("category") == "cross_document"

        t0 = time.perf_counter()
        try:
            turn = await query_service.execute_query(
                query=q_text,
                session_id=session_id,
                retrieval_mode=RetrievalMode.HYBRID,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            if turn.grounding_status == GroundingStatus.EXECUTION_FAILED:
                execution_failures += 1
                if is_refusal_req:
                    refusal_evals.append(0)
                else:
                    citation_evals.append(0.0)
                    factual_evals.append(0)
                    claim_evals.append(0)
                if is_calc_req:
                    calculator_evals.append(0)
                if is_cross_doc:
                    cross_doc_evals.append(0.0)

                itemized_results.append({
                    "query_id": qid,
                    "query": q_text,
                    "category": item.get("category"),
                    "error": "Turn returned EXECUTION_FAILED",
                    "success": False,
                    "latency_ms": elapsed_ms,
                })
                continue

            tools_called = [rec.tool_name for rec in turn.tool_executions]

            # 1. Refusal evaluation
            if is_refusal_req:
                refused = is_refusal_answer(turn.canonical_answer, turn.grounding_status)
                refusal_evals.append(1 if refused else 0)
            else:
                # 2. Citation validity evaluation (supported queries MUST provide valid citations)
                if not turn.citations:
                    # Missing citations on supported queries receive 0.0 (NO FREE PASS)
                    citation_evals.append(0.0)
                else:
                    retrieved_chunk_ids = {c.chunk.chunk_id for c in turn.evidence}
                    retrieved_doc_ids = {c.chunk.document_id.casefold() for c in turn.evidence}
                    valid_citations = 0
                    for c_tag in turn.citations:
                        match = re.search(r"\[([a-zA-Z0-9_-]+)#([a-zA-Z0-9_-]+)\]", c_tag)
                        if match:
                            doc_tag = match.group(1).casefold()
                            chk_tag = match.group(2)
                            if chk_tag in retrieved_chunk_ids or doc_tag in retrieved_doc_ids:
                                valid_citations += 1
                    citation_evals.append(valid_citations / len(turn.citations))

                # 3. Factual answer correctness
                fact_correct = compute_factual_correctness(
                    answer=turn.canonical_answer,
                    expected=item.get("expected_answer", ""),
                    evidence_spans=item.get("evidence_spans", []),
                )
                factual_evals.append(1 if fact_correct else 0)

                # 4. Claim-to-evidence support
                claim_supp = verify_claim_evidence_support(turn.canonical_answer, turn.evidence)
                claim_evals.append(1 if claim_supp else 0)

            # 5. Calculator invocation evaluation
            if is_calc_req:
                calc_invoked = "calculate" in tools_called
                num_correct = check_numerical_match(turn.canonical_answer, item.get("expected_answer", ""))
                calculator_evals.append(1 if (calc_invoked and num_correct) else 0)

            # 6. Cross-document synthesis coverage
            if is_cross_doc:
                rel_docs = [Path(d).stem.casefold() for d in item.get("relevant_doc_ids", [])]
                cited_docs = set()
                for c_tag in turn.citations:
                    match = re.search(r"\[([a-zA-Z0-9_-]+)#", c_tag)
                    if match:
                        cited_docs.add(match.group(1).casefold())
                cov = sum(1 for d in rel_docs if d in cited_docs) / len(rel_docs) if rel_docs else 1.0
                cross_doc_evals.append(cov)

            itemized_results.append({
                "query_id": qid,
                "query": q_text,
                "category": item.get("category"),
                "answer": turn.canonical_answer,
                "grounding_status": turn.grounding_status.value,
                "tools_executed": tools_called,
                "citations": turn.citations,
                "latency_ms": elapsed_ms,
                "success": True,
            })

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            execution_failures += 1
            if is_refusal_req:
                refusal_evals.append(0)
            else:
                citation_evals.append(0.0)
                factual_evals.append(0)
                claim_evals.append(0)
            if is_calc_req:
                calculator_evals.append(0)
            if is_cross_doc:
                cross_doc_evals.append(0.0)

            itemized_results.append({
                "query_id": qid,
                "query": q_text,
                "category": item.get("category"),
                "error": f"{type(exc).__name__}: {exc}",
                "success": False,
                "latency_ms": elapsed_ms,
            })

    # Multi-turn interaction test
    multi_turn_success = False
    try:
        mt_session = f"eval_multiturn_{int(time.time())}"
        t1 = await query_service.execute_query(
            query="What is ADR 001?",
            session_id=mt_session,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        t2 = await query_service.execute_query(
            query="What was the reason for rejecting Milvus?",
            session_id=mt_session,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        multi_turn_success = (
            t1.grounding_status != GroundingStatus.EXECUTION_FAILED
            and t2.grounding_status != GroundingStatus.EXECUTION_FAILED
            and ("milvus" in t2.canonical_answer.casefold() or "memory" in t2.canonical_answer.casefold() or "ram" in t2.canonical_answer.casefold())
        )
    except Exception:
        multi_turn_success = False

    metrics_dict = {
        "factual_answer_correctness": {
            "numerator": sum(factual_evals),
            "denominator": len(factual_evals),
            "value": (sum(factual_evals) / len(factual_evals)) if factual_evals else None,
        },
        "claim_support_rate": {
            "numerator": sum(claim_evals),
            "denominator": len(claim_evals),
            "value": (sum(claim_evals) / len(claim_evals)) if claim_evals else None,
        },
        "citation_validity_rate": {
            "numerator": sum(citation_evals),
            "denominator": len(citation_evals),
            "value": (sum(citation_evals) / len(citation_evals)) if citation_evals else None,
        },
        "unsupported_refusal_rate": {
            "numerator": sum(refusal_evals),
            "denominator": len(refusal_evals),
            "value": (sum(refusal_evals) / len(refusal_evals)) if refusal_evals else None,
        },
        "calculator_usage_rate": {
            "numerator": sum(calculator_evals),
            "denominator": len(calculator_evals),
            "value": (sum(calculator_evals) / len(calculator_evals)) if calculator_evals else None,
        },
        "cross_document_coverage": {
            "numerator": sum(cross_doc_evals),
            "denominator": len(cross_doc_evals),
            "value": (sum(cross_doc_evals) / len(cross_doc_evals)) if cross_doc_evals else None,
        },
        "multi_turn_correctness": {
            "numerator": 1 if multi_turn_success else 0,
            "denominator": 1,
            "value": 1.0 if multi_turn_success else 0.0,
        },
        "latency": {
            "p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
            "p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "mean_ms": float(np.mean(latencies)) if latencies else None,
        },
    }

    summary = {
        "status": "VERIFIED_LIVE",
        "model_id": active_model,
        "execution_failures": execution_failures,
        "total_queries": len(dataset),
        "successful_turns": sum(1 for r in itemized_results if r.get("success")),
        "metrics": metrics_dict,
        "itemized_results": itemized_results,
    }

    gate_status, violations = evaluate_generation_gate(summary)
    summary["gate_status"] = gate_status
    summary["violations"] = violations

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nFinal Generation Gate Status: [{gate_status}]")
    if violations:
        print("Gate Violations:")
        for v in violations:
            print(f"  - {v}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Real Answer-Level Generation Evaluation Harness")
    parser.add_argument("--dataset", type=str, default="labs/lab07_final_project/evaluation/datasets/final_acceptance_holdout_50.json")
    parser.add_argument("--output", type=str, default="labs/lab07_final_project/evaluation/baselines/generation_results.json")
    parser.add_argument("--llm-url", type=str, default="http://localhost:1234/v1")
    parser.add_argument("--model-id", type=str, default="ministral-3b")
    args = parser.parse_args()

    asyncio.run(run_generation_evaluation(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        llm_url=args.llm_url,
        model_id=args.model_id,
    ))


if __name__ == "__main__":
    main()
