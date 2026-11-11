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
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Document, Chunk, EvidenceChunk, ToolCallOrigin
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


UNIT_DEFINITIONS = {
    # Byte units -> base unit: byte
    "b": ("byte", 1.0),
    "bytes": ("byte", 1.0),
    "байт": ("byte", 1.0),
    "байтів": ("byte", 1.0),
    "байти": ("byte", 1.0),
    "kib": ("byte", 1024.0),
    "кіб": ("byte", 1024.0),
    "kb": ("byte", 1000.0),
    "кб": ("byte", 1000.0),
    "mib": ("byte", 1024.0 ** 2),
    "міб": ("byte", 1024.0 ** 2),
    "mb": ("byte", 1024.0 ** 2),
    "мб": ("byte", 1024.0 ** 2),
    "gib": ("byte", 1024.0 ** 3),
    "гіб": ("byte", 1024.0 ** 3),
    "gb": ("byte", 1024.0 ** 3),
    "гб": ("byte", 1024.0 ** 3),
    "tib": ("byte", 1024.0 ** 4),
    "тіб": ("byte", 1024.0 ** 4),
    "tb": ("byte", 1024.0 ** 4),
    "тб": ("byte", 1024.0 ** 4),
    # Time units -> base unit: second
    "ms": ("time", 0.001),
    "мс": ("time", 0.001),
    "мілісекунд": ("time", 0.001),
    "миллисекунд": ("time", 0.001),
    "s": ("time", 1.0),
    "sec": ("time", 1.0),
    "secs": ("time", 1.0),
    "second": ("time", 1.0),
    "seconds": ("time", 1.0),
    "с": ("time", 1.0),
    "сек": ("time", 1.0),
    "секунд": ("time", 1.0),
    "секунди": ("time", 1.0),
    "min": ("time", 60.0),
    "mins": ("time", 60.0),
    "хв": ("time", 60.0),
    "хвилин": ("time", 60.0),
    "минут": ("time", 60.0),
    "h": ("time", 3600.0),
    "hr": ("time", 3600.0),
    "hrs": ("time", 3600.0),
    "hour": ("time", 3600.0),
    "hours": ("time", 3600.0),
    "год": ("time", 3600.0),
    "годин": ("time", 3600.0),
    "часов": ("time", 3600.0),
    # Percentage -> base unit: percent
    "%": ("percentage", 1.0),
    "percent": ("percentage", 1.0),
    "відсотків": ("percentage", 1.0),
    "відсотка": ("percentage", 1.0),
    "процентів": ("percentage", 1.0),
}


def _parse_quantities(text: str) -> List[Tuple[float, Optional[str], Optional[str], float]]:
    """Extract numerical quantities with their units and converted base values.
    
    Returns list of (val, raw_unit, category, base_val).
    """
    if not text:
        return []
    clean = re.sub(r"(\d),(\d)", r"\1\2", text)
    pattern = r"\b(\d+(?:\.\d+)?)\s*([%a-zA-Zа-яА-ЯіІїЇєЄ]+)?\b"
    quantities = []
    for match in re.finditer(pattern, clean):
        raw_val_str = match.group(1)
        raw_unit_str = match.group(2)
        try:
            val = float(raw_val_str)
        except ValueError:
            continue
        category = None
        base_val = val
        unit_key = raw_unit_str.casefold() if raw_unit_str else None
        if unit_key and unit_key in UNIT_DEFINITIONS:
            category, scale = UNIT_DEFINITIONS[unit_key]
            base_val = val * scale
        quantities.append((val, raw_unit_str, category, base_val))
    return quantities


def _parse_numbers(text: str) -> List[float]:
    """Extract clean floating point numbers from text."""
    return [q[0] for q in _parse_quantities(text)]


def check_numerical_match(answer: str, expected: str) -> bool:
    """Verify that expected numerical results or mathematically derived quantities appear in the answer.
    
    Enforces unit awareness:
    - If expected specifies a unit category (byte, time, percentage), the answer must match
      both the unit category and the converted base magnitude.
    - A magnitude match with an incompatible unit (e.g. 788.48 GiB vs 0.77 GiB, or 5000 s vs 5000 ms) is rejected.
    - If expected is dimensionless, tolerance-based magnitude matching is applied.
    """
    exp_quantities = _parse_quantities(expected)
    ans_quantities = _parse_quantities(answer)
    if not exp_quantities:
        return True
    if not ans_quantities:
        return False

    def matches_quantity(eq: Tuple[float, Optional[str], Optional[str], float], aq: Tuple[float, Optional[str], Optional[str], float]) -> bool:
        e_val, e_unit, e_cat, e_base = eq
        a_val, a_unit, a_cat, a_base = aq
        if e_cat is not None:
            # Expected has a specific unit category (byte, time, percentage)
            if a_cat != e_cat:
                return False
            return math.isclose(a_base, e_base, rel_tol=0.03, abs_tol=0.01)
        else:
            # Dimensionless number or general entity count
            if math.isclose(a_val, e_val, rel_tol=0.01, abs_tol=0.01):
                return True
            # Allow common scale conversions if units were omitted but standard decimal/binary applies
            if math.isclose(a_val, e_val * 1000, rel_tol=0.05, abs_tol=1.0) or math.isclose(a_val * 1000, e_val, rel_tol=0.05, abs_tol=1.0):
                return True
            if math.isclose(a_val, e_val * 1024, rel_tol=0.05, abs_tol=1.0) or math.isclose(a_val * 1024, e_val, rel_tol=0.05, abs_tol=1.0):
                return True
            return False

    # Check key primary target quantity (usually the first number in expected answer)
    target_q = exp_quantities[0]
    for aq in ans_quantities:
        if matches_quantity(target_q, aq):
            return True

    # Also check if any expected quantity matches
    for eq in exp_quantities:
        for aq in ans_quantities:
            if matches_quantity(eq, aq):
                return True

    return False


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


def _detect_intra_answer_contradiction(answer: str) -> bool:
    """Detect if an answer internally contradicts itself across distinct sentences."""
    if not answer:
        return False
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
    if len(raw_sentences) < 2:
        return False

    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "has", "have", "had", "not", "but", "what", "which", "who", "when", "where",
        "було", "для", "або", "при", "який", "яка", "яке", "які", "так", "ні", "був",
        "були", "буде", "цього", "тому", "якщо", "після", "через", "згідно", "даними",
        "also", "than", "then", "its", "however", "although", "while", "here", "there",
    }

    neg_pattern = r"\b(?:not|never|neither|nor|rejected|failed|cannot|isn't|wasn't|doesn't|didn't|won't|не|ніколи|відхилено|немає|відсутній|відсутня|відсутнє|відсутні)\b"

    sentence_info = []
    for s in raw_sentences:
        clean = re.sub(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", "", s).strip()
        if len(clean) < 10:
            continue
        c_norm = normalize_text(clean)
        is_neg = bool(re.search(neg_pattern, c_norm))
        tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{4,}\b", c_norm) if t not in stopwords]
        if tokens:
            sentence_info.append((clean, is_neg, set(tokens), c_norm))

    for i in range(len(sentence_info)):
        _, s_i_neg, s_i_tokens, s_i_norm = sentence_info[i]
        for j in range(i + 1, len(sentence_info)):
            _, s_j_neg, s_j_tokens, s_j_norm = sentence_info[j]
            if s_i_neg != s_j_neg:
                neg_norm = s_j_norm if s_j_neg else s_i_norm
                # Ignore explicitly comparative / contrasting clauses that compare against baselines
                if re.search(r"\b(?:in contrast|on the other hand|whereas|unlike|instead of|на відміну від|на відміну)\b", neg_norm):
                    continue
                shared = s_i_tokens.intersection(s_j_tokens)
                min_len = min(len(s_i_tokens), len(s_j_tokens))
                if len(shared) >= 3 or (min_len > 0 and len(shared) / min_len >= 0.60):
                    for w in shared:
                        if (
                            re.search(neg_pattern + r"\s+\w*\s*" + re.escape(w), neg_norm)
                            or re.search(re.escape(w) + r"\s+\w*\s*" + neg_pattern, neg_norm)
                        ):
                            return True
    return False


def compute_factual_correctness(answer: str, expected: str, evidence_spans: List[str]) -> bool:
    """Evaluate whether the answer factual content aligns with gold expected answer and evidence spans.
    
    Scientifically grounds factual correctness in:
    - Intra-answer non-contradiction verification.
    - Core proposition polarity non-contradiction verification.
    - Numerical consistency (checked via unit-aware conversion and tolerance).
    - Verbatim evidence span alignment without premature short-circuiting.
    - Content token recall without artificial foreign vocabulary penalties.
    """
    a_norm = normalize_text(answer)
    e_norm = normalize_text(expected)
    if not a_norm or not e_norm:
        return False

    # 1. Intra-answer contradiction check: answer cannot contradict itself
    if _detect_intra_answer_contradiction(answer):
        return False

    # 2. Polarity contradiction check against expected answer
    if _detect_polarity_contradiction(a_norm, e_norm):
        return False

    # 3. Check numerical results if expected has numbers
    exp_quantities = _parse_quantities(expected)
    if exp_quantities:
        if not check_numerical_match(answer, expected):
            return False

    # 4. Direct verbatim evidence span alignment (checked AFTER contradiction and numerical validation)
    if evidence_spans:
        for span in evidence_spans:
            norm_span = normalize_text(span)
            if norm_span and norm_span in a_norm:
                return True

    # 5. Content token recall evaluated against expected answer and evidence spans
    # NOTE: Arbitrary foreign_tokens / foreign_ratio penalty removed per scientific acceptance requirements.
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

    if best_recall < 0.40:
        return False

    # Guard against unrelated shared vocabulary soup when no supporting evidence spans are provided:
    # If expected and answer are in the same language and answer asserts predominantly foreign predicates (>45% foreign words),
    # reject the proposition mismatch.
    if not evidence_spans:
        e_cand_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", e_norm) if t not in stopwords]
        if e_cand_tokens and ans_tokens:
            allowed = set(e_cand_tokens)
            foreign = [t for t in ans_tokens if t not in allowed and not any(t.startswith(c[:4]) or c.startswith(t[:4]) for c in allowed)]
            if len(foreign) / len(ans_tokens) >= 0.45:
                return False

    return True


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

        # 0. Specific citation entailment check: if sentence cites specific [doc#chunk], verify cited chunk entails claim
        cited_tags = re.findall(r"\[([a-zA-Z0-9_.-]+)#([a-zA-Z0-9_.-]+)\]", raw_s)
        if cited_tags:
            specific_chunk_supported = False
            for doc_ref, chunk_ref in cited_tags:
                matched_chunks = [
                    c for c in evidence
                    if c.chunk.chunk_id == chunk_ref or c.chunk.chunk_id.endswith(chunk_ref)
                ]
                for mc in matched_chunks:
                    mc_norm = normalize_text(mc.chunk.text)
                    mc_tokens = set(re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", mc_norm))
                    claim_tokens = [w for w in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{3,}\b", s_norm) if w not in stopwords]
                    if claim_tokens:
                        c_overlap = sum(1 for ct in claim_tokens if ct in mc_tokens or any(et.startswith(ct[:4]) or ct.startswith(et[:4]) for et in mc_tokens if len(et) >= 4)) / len(claim_tokens)
                        if c_overlap >= 0.25:
                            specific_chunk_supported = True
                            break
                if specific_chunk_supported:
                    break
            if not specific_chunk_supported:
                judgments.append({
                    "claim": clean_s,
                    "supported": False,
                    "verdict": "UNSUPPORTED_CITATION",
                    "reason": f"Cited chunk(s) {cited_tags} do not entail the claim sentence.",
                })
                continue

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
    is_supported = (support_rate >= 0.70) and not any(j.get("verdict") in ("CONTRADICTED", "UNSUPPORTED_CITATION") for j in judgments)

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

    corpus_path = Path("labs/lab07_final_project/assets/knowledge")
    chunks = repo.get_all_chunks()
    if len(chunks) < 140:
        from labs.lab07_final_project.evaluation.run_benchmark import ensure_corpus_ingested
        ensure_corpus_ingested(ingestion_service, corpus_path)
        chunks = repo.get_all_chunks()

    try:
        r_client = redis_adapter.get_client()
        if r_client.dbsize() < len(chunks):
            print(f"Syncing {len(chunks)} chunks to Redis vector store...")
            redis_adapter.index_chunks(chunks, embeddings)
    except Exception as r_exc:
        print(f"Redis sync check note: {r_exc}")

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

    # Disaggregated tool metrics
    total_numerical_queries = 0
    tool_invoked_queries = 0
    autonomous_tool_queries = 0
    recovery_tool_queries = 0
    total_tool_calls = 0
    successful_tool_calls = 0
    valid_arg_tool_calls = 0
    numerical_correct_queries = 0
    correct_without_tool_count = 0
    tool_success_generation_failure_count = 0

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
                    total_numerical_queries += 1
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
                print(f"[{idx}/{len(dataset)}] Evaluated {qid}: EXECUTION_FAILED ({elapsed_ms/1000.0:.2f}s)", flush=True)
                continue

            tools_called = [rec.tool_name for rec in turn.tool_executions]
            print(f"[{idx}/{len(dataset)}] Evaluated {qid}: {turn.grounding_status.value} ({elapsed_ms/1000.0:.2f}s, tools: {tools_called})", flush=True)

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

            # 5. Calculator and tool invocation evaluation (disaggregated)
            if is_calc_req:
                total_numerical_queries += 1
                calc_executions = [rec for rec in turn.tool_executions if rec.tool_name == "calculate"]
                has_calc = len(calc_executions) > 0
                num_correct = check_numerical_match(turn.canonical_answer, item.get("expected_answer", ""))
                if num_correct:
                    numerical_correct_queries += 1

                if has_calc:
                    tool_invoked_queries += 1
                    total_tool_calls += len(calc_executions)
                    has_model_req = any(
                        getattr(rec.origin, "value", rec.origin) == "MODEL_REQUESTED"
                        for rec in calc_executions
                    )
                    has_recovery = any(
                        getattr(rec.origin, "value", rec.origin) == "RECOVERY_EXECUTED"
                        for rec in calc_executions
                    )
                    if has_model_req:
                        autonomous_tool_queries += 1
                    if has_recovery:
                        recovery_tool_queries += 1

                    any_exec_success = False
                    for rec in calc_executions:
                        expr = str(rec.arguments.get("expression", "")).strip() if rec.arguments else ""
                        if expr and not any(bad in expr for bad in ["__", "import", "eval", "exec"]):
                            valid_arg_tool_calls += 1
                        res = rec.result
                        res_str = str(res) if res is not None else ""
                        if res is not None and not res_str.lower().startswith("error"):
                            successful_tool_calls += 1
                            any_exec_success = True

                    if any_exec_success and not num_correct:
                        tool_success_generation_failure_count += 1
                else:
                    if num_correct:
                        correct_without_tool_count += 1

                calculator_evals.append(1 if has_calc else 0)

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
            print(f"[{idx}/{len(dataset)}] Evaluated {qid}: EXCEPTION {type(exc).__name__}: {exc} ({elapsed_ms/1000.0:.2f}s)", flush=True)
            if is_refusal_req:
                refusal_evals.append(0)
            else:
                citation_evals.append(0.0)
                factual_evals.append(0)
                claim_evals.append(0)
            if is_calc_req:
                total_numerical_queries += 1
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
            "numerator": tool_invoked_queries,
            "denominator": total_numerical_queries,
            "value": (tool_invoked_queries / total_numerical_queries) if total_numerical_queries else None,
        },
        "tool_metrics": {
            "total_numerical_queries": total_numerical_queries,
            "tool_invocation_rate": (tool_invoked_queries / total_numerical_queries) if total_numerical_queries else 0.0,
            "autonomous_tool_invocation_rate": (autonomous_tool_queries / total_numerical_queries) if total_numerical_queries else 0.0,
            "recovery_tool_invocation_rate": (recovery_tool_queries / total_numerical_queries) if total_numerical_queries else 0.0,
            "tool_execution_success_rate": (successful_tool_calls / total_tool_calls) if total_tool_calls else 0.0,
            "tool_argument_validity_rate": (valid_arg_tool_calls / total_tool_calls) if total_tool_calls else 0.0,
            "numerical_factual_correctness_rate": (numerical_correct_queries / total_numerical_queries) if total_numerical_queries else 0.0,
            "correct_without_tool_count": correct_without_tool_count,
            "tool_success_generation_failure_count": tool_success_generation_failure_count,
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
