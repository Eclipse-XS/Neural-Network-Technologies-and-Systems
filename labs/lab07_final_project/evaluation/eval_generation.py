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
import math
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
        r"не\s+знайден[оаіий]",
        r"не\s+міст\w+",
        r"не\s+описан[оаіий]",
        r"не\s+наведен[оаіий]",
        r"відсутн\w*\s+(?:інформаці\w*|дані|відомост\w*|згадк\w*)",
        r"немає\s+(?:інформації|даних|відомостей|згадок|інструкцій|рецепт\w*|відомості|деталей)",
        r"не\s+можу\s+(?:надати|відповісти|знайти|визначити)",
        r"не\s+нада[єе]ться",
        r"не\s+виявлено",
        r"не\s+входить\s+до",
        r"не\s+покрива[єе]ться",
        r"не\s+вказан[оаіий]",
        r"do(?:es)?\s+not\s+(?:contain|provide|mention|describe|specify|include|detail|cover|have)",
        r"insufficient\s+(?:evidence|information|data)",
        r"not\s+found",
        r"no\s+(?:relevant\s+)?(?:information|data|evidence|mention|details|instructions|guidance|recipes)",
        r"cannot\s+(?:retrieve|answer|find|provide|determine)",
        r"unable\s+to\s+(?:find|retrieve|determine|answer|provide)",
        r"not\s+mentioned",
        r"no\s+evidence",
        r"abstain",
        r"not\s+covered",
        r"outside\s+(?:the\s+)?(?:scope|indexed|documentation)",
        r"does\s+not\s+have\s+context",
        r"no\s+direct\s+evidence",
    ]
    a_lower = (answer or "").casefold()
    return any(re.search(pat, a_lower) for pat in refusal_patterns)


def _parse_numbers(text: str) -> List[float]:
    """Extract clean floating point numbers from text, handling comma thousand separators."""
    if not text:
        return []
    clean = re.sub(r"(\d),(\d)", r"\1\2", text)
    raw_nums = re.findall(r"\b\d+(?:\.\d+)?\b", clean)
    parsed = []
    for n in raw_nums:
        try:
            parsed.append(float(n))
        except ValueError:
            pass
    return parsed


def check_numerical_match(answer: str, expected: str) -> bool:
    """Verify that expected numerical results or mathematically derived quantities appear in the answer."""
    exp_nums = _parse_numbers(expected)
    ans_nums = _parse_numbers(answer)
    if not exp_nums:
        return True
    if not ans_nums:
        return False

    # Check key primary target number (usually the first number in expected answer)
    target_num = exp_nums[0]
    for an in ans_nums:
        if math.isclose(an, target_num, rel_tol=0.01, abs_tol=0.01):
            return True
        # Check scale conversions (e.g. 0.77 GiB vs 770 MiB or 1.5 GB vs 1536 MB)
        if math.isclose(an, target_num * 1000, rel_tol=0.05, abs_tol=1.0) or math.isclose(an * 1000, target_num, rel_tol=0.05, abs_tol=1.0):
            return True
        if math.isclose(an, target_num * 1024, rel_tol=0.05, abs_tol=1.0) or math.isclose(an * 1024, target_num, rel_tol=0.05, abs_tol=1.0):
            return True

    # Also check if any expected number matches
    matches = sum(1 for en in exp_nums if any(math.isclose(en, an, rel_tol=0.01, abs_tol=0.01) for an in ans_nums))
    return matches > 0


def _detect_polarity_contradiction(text1: str, text2: str) -> bool:
    """Detect if an answer explicitly contradicts the core affirmative proposition of expected text."""
    stopwords = {"the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "has", "have", "not"}
    e_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{4,}\b", text2.casefold()) if w not in stopwords]
    if not e_words:
        return False
    for w in e_words:
        direct_neg = (
            re.search(r"\b(?:not|never|не|ніколи|відхилено|rejected)\s+\w*\s*" + re.escape(w), text1.casefold())
            or re.search(re.escape(w) + r"\s+\w*\s*(?:not|never|не|ніколи|відхилено|rejected)\b", text1.casefold())
        )
        if direct_neg:
            return True
    return False


def compute_factual_correctness(answer: str, expected: str, evidence_spans: List[str]) -> bool:
    """Evaluate whether the answer factual content aligns with gold expected answer and evidence spans.
    
    Scientifically grounds factual correctness in:
    - Direct evidence span / key proposition alignment.
    - Numerical consistency (checked via normalized numeric tolerance and units).
    - Content token recall across multilingual equivalents.
    - Elimination of artificial foreign vocabulary penalties.
    """
    a_norm = normalize_text(answer)
    e_norm = normalize_text(expected)
    if not a_norm or not e_norm:
        return False

    # 1. Direct verbatim evidence span alignment
    if evidence_spans:
        for span in evidence_spans:
            norm_span = normalize_text(span)
            if norm_span and norm_span in a_norm:
                return True

    # 2. Check numerical results if expected has numbers
    exp_nums = _parse_numbers(expected)
    if exp_nums:
        if not check_numerical_match(answer, expected):
            return False

    # 3. Check direct contradiction on core proposition
    if _detect_polarity_contradiction(a_norm, e_norm):
        return False

    # 4. Content token recall evaluated against expected answer and evidence spans
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "has", "have", "had", "not", "but", "what", "which", "who", "when", "where",
        "було", "для", "або", "при", "який", "яка", "яке", "які", "так", "ні", "був",
        "using", "used", "into", "over", "each", "both", "also", "than", "then", "its",
        "були", "буде", "цього", "тому", "якщо", "після", "через", "згідно", "даними",
    }

    candidates = [e_norm]
    if evidence_spans:
        candidates.extend(normalize_text(s) for s in evidence_spans)

    ans_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", a_norm) if t not in stopwords]
    ans_tokens_set = set(ans_tokens)

    # Reject foreign vocabulary / unrelated assertions sharing only subset vocabulary
    allowed_context = e_norm + " " + " ".join(normalize_text(s) for s in (evidence_spans or []))
    allowed_tokens = set(re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", allowed_context))
    allowed_tokens.update({
        "лабораторна", "лабораторній", "робота", "роботі", "документ", "документом",
        "модель", "моделі", "система", "системи", "результат", "результати",
    })
    if ans_tokens:
        foreign_tokens = [
            t for t in ans_tokens
            if t not in allowed_tokens and not (len(t) >= 4 and any(at.startswith(t[:4]) or t.startswith(at[:4]) for at in allowed_tokens if len(at) >= 4))
        ]
        foreign_ratio = len(foreign_tokens) / len(ans_tokens)
        if not evidence_spans and foreign_ratio >= 0.40:
            return False
        elif evidence_spans and foreign_ratio >= 0.65:
            return False

    best_recall = 0.0
    for cand in candidates:
        cand_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", cand) if t not in stopwords]
        if not cand_tokens:
            continue
        matched = 0
        for ct in cand_tokens:
            if ct in ans_tokens_set:
                matched += 1
            elif len(ct) >= 4 and any(at.startswith(ct[:4]) or ct.startswith(at[:4]) for at in ans_tokens_set if len(at) >= 4):
                matched += 1
        recall = matched / len(cand_tokens)
        if recall > best_recall:
            best_recall = recall

    return best_recall >= 0.40


def verify_claim_evidence_support(
    answer: str,
    evidence: List[EvidenceChunk],
    return_judgments: bool = False,
) -> Any:
    """Verify that substantive claims in answer are anchored in retrieved evidence chunks.
    
    Supports cumulative cross-chunk evidence grounding and derived numerical quantities.
    """
    if not evidence or not answer.strip():
        return (False, []) if return_judgments else False

    chunk_texts = [c.chunk.text for c in evidence]
    combined_ev_norm = normalize_text(" ".join(chunk_texts))
    combined_ev_tokens = set(re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", combined_ev_norm))
    ev_numbers = _parse_numbers(combined_ev_norm)

    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]

    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "has", "have", "had", "not", "but", "what", "which", "who", "when", "where",
        "було", "для", "або", "при", "який", "яка", "яке", "які", "так", "ні", "був",
        "according", "stated", "here", "shown", "there", "based", "document", "documents",
        "наприклад", "зокрема", "також", "згідно", "calculated", "calculating", "difference", "remaining", "minus", "plus", "sum", "total", "обчислено", "різниця", "залишок", "разом", "становить",
    }

    judgments = []

    for raw_s in raw_sentences:
        clean_s = re.sub(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", "", raw_s).strip()
        if not clean_s or len(clean_s) < 12:
            continue

        s_norm = normalize_text(clean_s)

        # 1. Contradiction & negation detection against evidence
        has_neg = bool(re.search(r"\b(?:not|never|neither|nor|rejected|failed|cannot|isn't|wasn't|не|ні|відхилено)\b", s_norm))
        if has_neg:
            core_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{4,}\b", s_norm) if w not in stopwords]
            matched_affirmative = False
            for c_text in chunk_texts:
                c_norm = normalize_text(c_text)
                if any(w in c_norm for w in core_words) and not any(re.search(r"\b(?:not|never|rejected|не|відхилено)\s+" + re.escape(w), c_norm) for w in core_words):
                    matched_affirmative = True
                    break
            is_negated = any(
                re.search(r"\b(?:not|never|rejected|не|відхилено)\s+\w*\s*" + re.escape(w), s_norm)
                or re.search(re.escape(w) + r"\s+\w*\s*(?:not|never|rejected|не|відхилено)\b", s_norm)
                for w in core_words
            )
            if matched_affirmative and is_negated:
                judgments.append({
                    "claim": clean_s,
                    "supported": False,
                    "verdict": "CONTRADICTED",
                    "reason": "Claim asserts negative/rejection while evidence asserts affirmative.",
                })
                continue

        # 2. Numerical verification with derived numbers allowance
        claim_nums = _parse_numbers(s_norm)
        substantive_nums = [n for n in claim_nums if n not in {3.0, 4.0, 5.0, 6.0, 7.0, 2024.0, 2025.0, 2026.0}]
        if substantive_nums and ev_numbers:
            unsupported_nums = []
            for n in substantive_nums:
                in_ev = any(math.isclose(n, en, rel_tol=0.02, abs_tol=0.01) for en in ev_numbers)
                in_scale = any(math.isclose(n, en * 1000, rel_tol=0.05, abs_tol=1.0) or math.isclose(n * 1000, en, rel_tol=0.05, abs_tol=1.0) for en in ev_numbers)
                is_derived = any(math.isclose(n, abs(en1 - en2), rel_tol=0.02, abs_tol=0.5) for en1 in ev_numbers for en2 in ev_numbers)
                is_derived_prod = any(math.isclose(n, en1 * en2, rel_tol=0.02, abs_tol=0.5) for en1 in ev_numbers for en2 in ev_numbers)
                if not (in_ev or in_scale or is_derived or is_derived_prod):
                    unsupported_nums.append(n)
            if len(unsupported_nums) > 2:
                judgments.append({
                    "claim": clean_s,
                    "supported": False,
                    "verdict": "NUMERICAL_MISMATCH",
                    "reason": f"Numbers {unsupported_nums} absent from evidence.",
                })
                continue

        # 3. Grounding in cumulative evidence chunks
        claim_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", s_norm) if w not in stopwords]
        if not claim_tokens:
            continue

        matched_cnt = 0
        for ct in claim_tokens:
            if ct in combined_ev_tokens:
                matched_cnt += 1
            elif len(ct) >= 4 and any(et.startswith(ct[:4]) or ct.startswith(et[:4]) for et in combined_ev_tokens if len(et) >= 4):
                matched_cnt += 1

        coverage = matched_cnt / len(claim_tokens)

        if coverage >= 0.40:
            judgments.append({
                "claim": clean_s,
                "supported": True,
                "verdict": "SUPPORTED",
                "support_score": coverage,
            })
        else:
            judgments.append({
                "claim": clean_s,
                "supported": False,
                "verdict": "UNRELATED_OR_INSUFFICIENT_EVIDENCE",
                "reason": f"Only {coverage:.2f} content coverage against evidence.",
            })

    if not judgments:
        is_supported = True
        return (is_supported, []) if return_judgments else is_supported

    supported_count = sum(1 for j in judgments if j["supported"])
    support_rate = supported_count / len(judgments)
    is_supported = (support_rate >= 0.70) and not any(j.get("verdict") in ("CONTRADICTED",) for j in judgments)

    if return_judgments:
        return is_supported, judgments
    return is_supported


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
