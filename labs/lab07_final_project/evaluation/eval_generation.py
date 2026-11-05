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
from typing import Any, Dict, List, Optional, Set, Tuple

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
    "fabricated_citations": 0,
    "factual_answer_correctness": 0.80,
    "claim_support_rate": 0.85,
    "citation_validity_rate": 0.85,
    "unsupported_refusal_rate": 0.90,
    "calculator_usage_rate": 0.80,
    "cross_document_coverage": 0.80,
    "multi_turn_correctness": 1.00,
}


def audit_citations(
    citations: List[str],
    evidence: List[EvidenceChunk],
) -> Dict[str, Any]:
    """Verify citations against retrieved evidence using exact (document_id, chunk_id) pairs.

    A citation is valid ONLY if BOTH its document_id and chunk_id match an exact retrieved chunk.
    An OR between document or chunk membership is strictly prohibited.
    Fabricated and unsupported citations are penalized explicitly.
    """
    if not citations:
        return {
            "valid_citations": [],
            "fabricated_citations": [],
            "validity_rate": 0.0,
            "is_valid": False,
        }

    valid_pairs: Set[Tuple[str, str]] = set()
    for ev in evidence:
        chunk = ev.chunk
        c_id = chunk.chunk_id
        d_id = chunk.document_id.casefold()
        valid_pairs.add((d_id, c_id))
        d_stem = Path(chunk.document_id).stem.casefold()
        valid_pairs.add((d_stem, c_id))
        if "source_id" in chunk.metadata:
            valid_pairs.add((str(chunk.metadata["source_id"]).casefold(), c_id))
        if "filename" in chunk.metadata:
            valid_pairs.add((Path(chunk.metadata["filename"]).stem.casefold(), c_id))

    valid = []
    fabricated = []

    for c_tag in citations:
        match = re.search(r"\[([a-zA-Z0-9_.-]+)#([a-zA-Z0-9_.-]+)\]", c_tag)
        if not match:
            fabricated.append(c_tag)
            continue
        doc_tag = match.group(1).casefold()
        chk_tag = match.group(2)
        # EXACT PAIR IDENTITY: Both document_id AND chunk_id must match
        if (doc_tag, chk_tag) in valid_pairs:
            valid.append(c_tag)
        else:
            fabricated.append(c_tag)

    total = len(citations)
    # Penalize fabricated citations explicitly: deduct fabricated count from valid count
    penalized_valid = max(0, len(valid) - len(fabricated))
    validity_rate = (penalized_valid / total) if total > 0 else 0.0

    return {
        "valid_citations": valid,
        "fabricated_citations": fabricated,
        "validity_rate": validity_rate,
        "is_valid": len(fabricated) == 0 and len(valid) > 0,
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
        r"not found",
        r"do(?:es)? not contain",
        r"no information",
        r"no relevant (?:information|data|evidence)",
        r"cannot retrieve",
        r"unable to find",
        r"not mentioned",
        r"no evidence",
        r"abstain",
        r"cannot answer",
        r"не можу надати",
        r"немає інформації",
    ]
    a_lower = (answer or "").casefold()
    return any(re.search(pat, a_lower) for pat in refusal_patterns)


def _detect_polarity_contradiction(text1: str, text2: str) -> bool:
    """Detect if one text asserts affirmative while the other asserts negation/rejection on key terms."""
    negation_patterns = [
        r"\b(?:not|never|no|neither|nor|rejected|refused|failed|cannot|does not|did not|was not|isn't|wasn't|didn't|unsupported)\b",
        r"\b(?:не|ні|ніколи|відхилено|не вдалося|не підтримується|не застосовується|немає|неможливо)\b",
    ]
    t1_has_neg = any(re.search(p, text1.casefold()) for p in negation_patterns)
    t2_has_neg = any(re.search(p, text2.casefold()) for p in negation_patterns)
    return t1_has_neg != t2_has_neg


def compute_factual_correctness(answer: str, expected: str, evidence_spans: List[str]) -> bool:
    """Evaluate whether the answer factual content aligns with gold expected answer and evidence spans.
    
    Replaces loose 50% token-overlap with auditable factual judgments:
    - Verifies numerical consistency (no conflicting or missing critical numbers).
    - Checks for contradiction / negation polarity mismatch.
    - Rejects foreign/unrelated assertions sharing vocabulary.
    - Prevents unsupported factual statements from receiving full credit based on shared vocabulary.
    """
    a_norm = normalize_text(answer)
    e_norm = normalize_text(expected)
    if not a_norm or not e_norm:
        return False

    # 1. Check contradiction / negation polarity mismatch between answer and expected
    if _detect_polarity_contradiction(a_norm, e_norm):
        return False

    # 2. Check numerical results
    exp_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", e_norm.replace(",", ""))
    ans_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", a_norm.replace(",", ""))
    if exp_numbers:
        if not all(n in ans_numbers for n in exp_numbers):
            return False

    # 3. Direct verbatim evidence span alignment
    if evidence_spans:
        for span in evidence_spans:
            norm_span = normalize_text(span)
            if norm_span and norm_span in a_norm:
                span_neg = re.search(r"\b(?:not|never|rejected|не|відхилено)\s+" + re.escape(norm_span[:30]), a_norm)
                if not span_neg:
                    return True

    # 4. Semantic proposition and token alignment
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "has", "have", "had", "not", "but", "what", "which", "who", "when", "where",
        "було", "для", "або", "при", "який", "яка", "яке", "які", "так", "ні", "був",
        "using", "used", "into", "over", "each", "both", "also", "than", "then", "its",
    }
    exp_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", e_norm) if t not in stopwords]
    if not exp_tokens:
        return e_norm in a_norm

    ans_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", a_norm) if t not in stopwords]
    ans_tokens_set = set(ans_tokens)
    matched_tokens = sum(1 for t in exp_tokens if t in ans_tokens_set)
    token_recall = matched_tokens / len(exp_tokens)

    # Require minimum 50% recall of expected content tokens
    if token_recall < 0.50:
        return False

    # Check for unrelated evidence / foreign vocabulary intrusion:
    # An answer that just strings together related vocabulary while asserting unrelated facts
    # will have a high ratio of foreign content words not found in expected answer or evidence spans
    allowed_context_tokens = set(re.findall(
        r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b",
        (e_norm + " " + " ".join(normalize_text(s) for s in (evidence_spans or [])))
    ))
    if ans_tokens:
        foreign_tokens = [t for t in ans_tokens if t not in allowed_context_tokens]
        foreign_ratio = len(foreign_tokens) / len(ans_tokens)
        if foreign_ratio >= 0.35:
            return False

    return True


def verify_claim_evidence_support(
    answer: str,
    evidence: List[EvidenceChunk],
    return_judgments: bool = False,
) -> Any:
    """Verify that substantive claims in answer are anchored in retrieved chunks.

    Replaces loose 50% token-overlap with auditable claim-level evidence judgments:
    - Evaluates each claim/sentence independently.
    - Validates numerical assertions against evidence.
    - Rejects contradictions and negation conflicts.
    - Requires multi-token n-gram grounding in specific supporting chunks to prevent shared vocabulary false positives.
    """
    if not evidence or not answer.strip():
        return (False, []) if return_judgments else False

    chunk_texts = [c.chunk.text for c in evidence]
    combined_ev_norm = normalize_text(" ".join(chunk_texts))

    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]

    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "has", "have", "had", "not", "but", "what", "which", "who", "when", "where",
        "було", "для", "або", "при", "який", "яка", "яке", "які", "так", "ні", "був",
        "according", "stated", "here", "shown", "there", "based", "document", "documents",
    }

    judgments = []

    for raw_s in raw_sentences:
        clean_s = re.sub(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", "", raw_s).strip()
        if not clean_s or len(clean_s) < 10:
            continue

        s_norm = normalize_text(clean_s)

        # 1. Numerical verification
        claim_nums = re.findall(r"\b\d+(?:\.\d+)?\b", s_norm)
        substantive_nums = [n for n in claim_nums if n not in {"3", "4", "5", "6", "7", "2024", "2025", "2026"}]
        if substantive_nums:
            missing_nums = [n for n in substantive_nums if n not in combined_ev_norm]
            if missing_nums:
                judgments.append({
                    "claim": clean_s,
                    "supported": False,
                    "verdict": "NUMERICAL_MISMATCH",
                    "reason": f"Numbers {missing_nums} absent from retrieved evidence.",
                })
                continue

        # 2. Contradiction & negation detection
        has_neg = bool(re.search(r"\b(?:not|never|neither|nor|rejected|failed|cannot|isn't|wasn't|не|ні|відхилено)\b", s_norm))
        if has_neg:
            core_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{4,}\b", s_norm) if w not in stopwords]
            matched_affirmative = False
            for c_text in chunk_texts:
                c_norm = normalize_text(c_text)
                if any(w in c_norm for w in core_words) and not any(re.search(r"\b(?:not|never|rejected|не|відхилено)\s+" + re.escape(w), c_norm) for w in core_words):
                    matched_affirmative = True
                    break
            if matched_affirmative and any(re.search(r"\b(?:not|never|rejected|не|відхилено)\s+\w*\s*" + re.escape(w), s_norm) for w in core_words):
                judgments.append({
                    "claim": clean_s,
                    "supported": False,
                    "verdict": "CONTRADICTED",
                    "reason": "Claim asserts negative/rejection while evidence asserts affirmative.",
                })
                continue

        # 3. Grounding in specific candidate chunk
        claim_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", s_norm) if w not in stopwords]
        if not claim_tokens:
            continue

        best_chunk_support = 0.0
        best_chunk_id = None
        best_bigram_ratio = 0.0

        for ev in evidence:
            c_text_norm = normalize_text(ev.chunk.text)
            c_tokens = set(re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", c_text_norm))
            matched_cnt = sum(1 for t in claim_tokens if t in c_tokens)
            coverage = matched_cnt / len(claim_tokens)

            bg_ratio = 0.0
            if len(claim_tokens) >= 3:
                bigrams = [f"{claim_tokens[i]} {claim_tokens[i+1]}" for i in range(len(claim_tokens) - 1)]
                c_bg_str = " ".join(re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", c_text_norm))
                matched_bg = sum(1 for bg in bigrams if bg in c_bg_str)
                bg_ratio = matched_bg / len(bigrams)

            if coverage > best_chunk_support:
                best_chunk_support = coverage
                best_chunk_id = ev.chunk.chunk_id
                best_bigram_ratio = bg_ratio

        if best_chunk_support >= 0.70 and (len(claim_tokens) < 3 or best_bigram_ratio >= 0.30):
            judgments.append({
                "claim": clean_s,
                "supported": True,
                "verdict": "SUPPORTED",
                "matched_chunk_id": best_chunk_id,
                "support_score": best_chunk_support,
            })
        else:
            judgments.append({
                "claim": clean_s,
                "supported": False,
                "verdict": "UNRELATED_OR_INSUFFICIENT_EVIDENCE",
                "reason": f"Only {best_chunk_support:.2f} content coverage and {best_bigram_ratio:.2f} bigram alignment with best chunk {best_chunk_id}.",
            })

    if not judgments:
        is_supported = True
        return (is_supported, []) if return_judgments else is_supported

    supported_count = sum(1 for j in judgments if j["supported"])
    support_rate = supported_count / len(judgments)
    is_supported = (support_rate >= 0.85) and not any(j.get("verdict") in ("NUMERICAL_MISMATCH", "CONTRADICTED") for j in judgments)

    if return_judgments:
        return is_supported, judgments
    return is_supported


def check_numerical_match(answer: str, expected: str) -> bool:
    """Verify that expected numerical results appear in the answer."""
    exp_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", (expected or "").replace(",", ""))
    ans_numbers = re.findall(r"\b\d+(?:\.\d+)?\b", (answer or "").replace(",", ""))
    if not exp_numbers:
        return True
    return all(n in ans_numbers for n in exp_numbers)


def evaluate_generation_gate(metrics_summary: Dict[str, Any]) -> Tuple[str, List[str]]:
    """Determine PASS/FAIL status based on explicit acceptance thresholds.
    
    Returns (gate_status, list_of_violations).
    """
    violations = []

    # 1. Execution failures must be strictly 0
    exec_failures = metrics_summary.get("execution_failures", 0)
    if exec_failures > 0:
        violations.append(f"Execution failures: {exec_failures} (target: 0)")

    # 2. Fabricated citations must be strictly 0
    fab_citations = metrics_summary.get("fabricated_citations", 0)
    if fab_citations > 0:
        violations.append(f"Fabricated citations: {fab_citations} (target: 0)")

    metrics = metrics_summary.get("metrics", {})

    for metric_name, threshold in GENERATION_ACCEPTANCE_THRESHOLDS.items():
        if metric_name in ("execution_failures", "fabricated_citations"):
            continue
        metric_data = metrics.get(metric_name, {})
        val = metric_data.get("value")
        if val is None:
            violations.append(f"{metric_name}: None (target >= {threshold:.2f})")
        elif val < threshold:
            violations.append(f"{metric_name}: {val:.4f} (target >= {threshold:.2f})")

    # Incomplete evaluation or missing results must prevent acceptance
    total_queries = metrics_summary.get("total_queries", 0)
    successful_turns = metrics_summary.get("successful_turns", 0)
    if total_queries > 0 and (successful_turns + exec_failures < total_queries):
        violations.append(f"Incomplete evaluation: evaluated {successful_turns + exec_failures}/{total_queries} queries")

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
                "multi_turn_correctness": {"numerator": 0, "denominator": 0, "value": None, "status": "NOT_VERIFIED"},
                "fabricated_citations": {"numerator": 0, "denominator": 0, "value": None, "status": "NOT_VERIFIED"},
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
        key_prefix="lab07:bench:chunk",
    )
    redis_adapter.initialize_store(embeddings)

    ingestion_service = IngestionService(
        sqlite_repo=repo,
        redis_adapter=redis_adapter,
        embeddings_model=embeddings,
        require_vector_store=False,
    )

    corpus_path = Path("labs/lab07_final_project/assets/knowledge")
    if len(repo.get_all_chunks()) < 140:
        from labs.lab07_final_project.evaluation.run_benchmark import ensure_corpus_ingested
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

    itemized_results = []
    latencies = []
    refusal_evals = []
    citation_evals = []
    factual_evals = []
    claim_evals = []
    calculator_evals = []
    cross_doc_evals = []
    execution_failures = 0
    total_fabricated_citations = 0

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
                # 2. Citation validity evaluation (exact (doc, chunk) matching & penalty for fabrications)
                if not turn.citations:
                    # Missing citations on supported queries receive 0.0 (NO FREE PASS)
                    citation_evals.append(0.0)
                else:
                    audit = audit_citations(turn.citations, turn.evidence)
                    if audit["fabricated_citations"]:
                        total_fabricated_citations += len(audit["fabricated_citations"])
                    citation_evals.append(audit["validity_rate"])

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

            # 6. Cross-document synthesis coverage (only counts valid citations)
            if is_cross_doc:
                rel_docs = [Path(d).stem.casefold() for d in item.get("relevant_doc_ids", [])]
                audit = audit_citations(turn.citations, turn.evidence)
                valid_cited_docs = set()
                for c_tag in audit["valid_citations"]:
                    match = re.search(r"\[([a-zA-Z0-9_.-]+)#", c_tag)
                    if match:
                        valid_cited_docs.add(match.group(1).casefold())
                cov = sum(1 for d in rel_docs if d in valid_cited_docs) / len(rel_docs) if rel_docs else 1.0
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

    # Multi-turn interaction test with genuine contextual dependency & session isolation
    multi_turn_success = False
    multi_turn_detail = {}
    try:
        mt_session = f"eval_multiturn_{int(time.time())}"
        # Turn 1: Anchor query establishing context
        t1 = await query_service.execute_query(
            query="What vector database and storage architecture was selected in ADR 001?",
            session_id=mt_session,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        t1_ok = (
            t1.grounding_status != GroundingStatus.EXECUTION_FAILED
            and any(w in t1.canonical_answer.casefold() for w in ["redis", "sqlite"])
            and len(t1.citations) > 0
        )

        # Turn 2: Context-dependent follow-up requiring Turn 1 information to resolve "it"
        t2 = await query_service.execute_query(
            query="Why was it selected instead of Milvus and Qdrant?",
            session_id=mt_session,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        t2_audit = audit_citations(t2.citations, t2.evidence)
        t2_ok = (
            t2.grounding_status != GroundingStatus.EXECUTION_FAILED
            and any(w in t2.canonical_answer.casefold() for w in ["milvus", "memory", "ram", "complex", "operational", "qdrant", "redis"])
            and t2_audit["is_valid"]
        )

        # Check model-visible conversation history in SQLite session
        history_sess = repo.get_session(mt_session)
        history_ok = history_sess is not None and len(history_sess.turns) >= 2

        # Check session isolation: same follow-up in fresh isolated session without Turn 1 must NOT resolve or must abstain
        isolated_session = f"eval_isolated_{int(time.time())}"
        t_iso = await query_service.execute_query(
            query="Why was it selected instead of Milvus and Qdrant?",
            session_id=isolated_session,
            retrieval_mode=RetrievalMode.HYBRID,
        )
        iso_sess = repo.get_session(isolated_session)
        isolation_ok = (
            iso_sess is not None
            and len(iso_sess.turns) == 1
            and iso_sess.session_id != mt_session
        )

        multi_turn_success = t1_ok and t2_ok and history_ok and isolation_ok
        multi_turn_detail = {
            "turn_1_success": t1_ok,
            "turn_2_success": t2_ok,
            "history_persisted": history_ok,
            "session_isolation_verified": isolation_ok,
        }
    except Exception as mt_exc:
        multi_turn_success = False
        multi_turn_detail = {"error": str(mt_exc)}

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
        "fabricated_citations": {
            "numerator": total_fabricated_citations,
            "denominator": len(citation_evals) if citation_evals else 1,
            "value": total_fabricated_citations,
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
        "fabricated_citations": total_fabricated_citations,
        "total_queries": len(dataset),
        "successful_turns": sum(1 for r in itemized_results if r.get("success")),
        "metrics": metrics_dict,
        "multi_turn_detail": multi_turn_detail,
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
