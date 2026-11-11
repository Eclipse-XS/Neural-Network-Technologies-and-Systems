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
from labs.lab07_final_project.src.knowledge_copilot.agents.tools import safe_calculate
from labs.lab07_final_project.src.knowledge_copilot.rag.embeddings import CopilotEmbeddings
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.llm_client import (
    LocalLLMClient,
    probe_lm_studio_readiness,
    LMStudioStatus,
    LMStudioProbeResult,
)
from labs.lab07_final_project.evaluation.metrics import normalize_text
from labs.lab07_final_project.src.knowledge_copilot.rag.lexical import (
    BILINGUAL_SYNONYMS,
    stem_word,
)
from labs.lab07_final_project.src.knowledge_copilot.agents.verifier import (
    KNOWN_TECHNOLOGIES,
    normalize_multi_citations,
    strip_markdown,
)

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
    """Evaluate whether an answer appropriately abstained on an unsupported query.

    Strips markdown styling so emphasis like 'do **not** explicitly' is parsed accurately,
    while strictly excluding partial premise substitutions that assert an unrelated model/tech.
    """
    if grounding_status == GroundingStatus.INSUFFICIENT_EVIDENCE:
        return True
    refusal_patterns = [
        r"не\s+знайден[оаіий]",
        r"не\s+міст\w+",
        r"не\s+(?:було\s+)?(?:прямо\s+)?(?:зазначен[оаіий]|наведен[оаіий]|описан[оаіий]|вказан[оаіий]|згадан[оаіий])",
        r"відсутн\w*\s+(?:інформаці\w*|дані|відомост\w*|згадк\w*)",
        r"(?:інформаці\w*|дані|відомост\w*|згадк\w*)\s+(?:відсутн\w*|немає)",
        r"немає\s+(?:жодної\s+|такої\s+|цієї\s+)?(?:інформації|даних|відомостей|згадок|інструкцій|рецепт\w*|відомості|деталей)",
        r"не\s+можу\s+(?:надати|відповісти|знайти|визначити)",
        r"не\s+нада[єе]ться",
        r"не\s+виявлено",
        r"не\s+входить\s+до",
        r"не\s+покрива[єе]ться",
        r"не\s+згадується",
        r"недостатньо\s+(?:даних|доказів|інформації)",
        r"do(?:es)?\s+not\s+(?:explicitly\s+)?(?:contain|provide|mention|describe|specify|include|detail|cover|have|report|list|state|configure)",
        r"(?:is|are)\s+not\s+(?:explicitly\s+)?(?:configured|mentioned|reported|listed|stated|documented|supported|covered)",
        r"not\s+(?:explicitly\s+)?(?:mentioned|listed|stated|reported|described|configured|documented|covered|found)",
        r"insufficient\s+(?:evidence|information|data)",
        r"no\s+(?:specific\s+|relevant\s+|direct\s+)?(?:information|data|evidence|mention|details|instructions|guidance|recipes|benchmark)",
        r"cannot\s+(?:retrieve|answer|find|provide|determine)",
        r"unable\s+to\s+(?:find|retrieve|determine|answer|provide)",
        r"\babstain\b",
        r"outside\s+(?:the\s+)?(?:scope|indexed|documentation)",
        r"does\s+not\s+have\s+context",
    ]
    a_lower = strip_markdown(answer or "").casefold()
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
    "секунда": ("time", 1.0),
    "секунду": ("time", 1.0),
    "min": ("time", 60.0),
    "mins": ("time", 60.0),
    "хв": ("time", 60.0),
    "хвилин": ("time", 60.0),
    "хвилини": ("time", 60.0),
    "хвилина": ("time", 60.0),
    "хвилину": ("time", 60.0),
    "минут": ("time", 60.0),
    "h": ("time", 3600.0),
    "hr": ("time", 3600.0),
    "hrs": ("time", 3600.0),
    "hour": ("time", 3600.0),
    "hours": ("time", 3600.0),
    "год": ("time", 3600.0),
    "годин": ("time", 3600.0),
    "години": ("time", 3600.0),
    "година": ("time", 3600.0),
    "годину": ("time", 3600.0),
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
    
    Supports comma thousands ('12,400') and space thousands ('12 400' / '3 072 000').
    Returns list of (val, raw_unit, category, base_val).
    """
    if not text:
        return []
    clean = re.sub(r"(\d),(\d)", r"\1\2", text)
    clean = re.sub(
        r"(?<!\d)(\d{1,3}(?:[ \u00a0]\d{3})+)(?!\d)",
        lambda m: m.group(1).replace(" ", "").replace("\u00a0", ""),
        clean,
    )
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

    # Prefix-ambiguous decimal/binary byte units (excludes strict IEC kib/mib/gib/tib)
    ambiguous_byte_exp = {
        "b": 0, "bytes": 0, "байт": 0, "байтів": 0, "байти": 0,
        "kb": 1, "кб": 1, "mb": 2, "мб": 2, "gb": 3, "гб": 3, "tb": 4, "тб": 4,
    }

    def matches_quantity(eq: Tuple[float, Optional[str], Optional[str], float], aq: Tuple[float, Optional[str], Optional[str], float]) -> bool:
        e_val, e_unit, e_cat, e_base = eq
        a_val, a_unit, a_cat, a_base = aq
        if e_cat is not None:
            # Expected has a specific unit category (byte, time, percentage)
            if a_cat != e_cat:
                return False
            if math.isclose(a_base, e_base, rel_tol=0.03, abs_tol=0.01):
                return True
            # Allow consistent decimal SI (1000^p) or binary (1024^p) interpretation ONLY when both units are prefix-ambiguous (e.g., 3.072 MB == 3,072,000 bytes)
            if e_cat == "byte":
                ek = (e_unit or "").casefold()
                ak = (a_unit or "").casefold()
                if ek in ambiguous_byte_exp and ak in ambiguous_byte_exp:
                    e_si = e_val * (1000.0 ** ambiguous_byte_exp[ek])
                    a_si = a_val * (1000.0 ** ambiguous_byte_exp[ak])
                    if math.isclose(a_si, e_si, rel_tol=0.03, abs_tol=0.01):
                        return True
                    e_iec = e_val * (1024.0 ** ambiguous_byte_exp[ek])
                    a_iec = a_val * (1024.0 ** ambiguous_byte_exp[ak])
                    if math.isclose(a_iec, e_iec, rel_tol=0.03, abs_tol=0.01):
                        return True
            return False
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

    # Identify primary target quantity: RHS of equation if '=' follows arithmetic expression, else first non-clock quantity
    exp_no_clock = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", expected)
    clean_exp_qs = _parse_quantities(exp_no_clock) or exp_quantities
    target_q = clean_exp_qs[0]
    if "=" in exp_no_clock:
        lhs, rhs = exp_no_clock.split("=", 1)
        if re.search(r"[+*/×-]", lhs):
            rhs_qs = _parse_quantities(rhs.split(")")[0])
            if rhs_qs:
                target_q = rhs_qs[0]

    for aq in ans_quantities:
        if matches_quantity(target_q, aq):
            return True

    return False


def _detect_polarity_contradiction(text1: str, text2: str) -> bool:
    """Detect if an answer explicitly contradicts the core affirmative proposition of expected text.

    Excludes function words, epistemic caveats ('not explicitly documented'), and cases
    where expected text itself shares the negative/rejected framing.
    """
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were", "has", "have", "not",
        "which", "what", "where", "when", "while", "into", "over", "under", "between", "through", "against",
        "system", "model", "data", "using", "state", "states", "stated", "mentioned", "documented",
        "configured", "available", "explicitly", "details", "mode", "design", "value", "values", "level",
        "система", "системи", "модель", "дані", "даними", "деталі", "інформація", "робота", "роботі",
        "звіт", "звіті", "було", "була", "були", "буде", "можна", "тільки", "лише", "після",
        "через", "згідно", "результаті", "також", "проте", "однак", "режим", "рівень",
    }
    t2_lower = strip_markdown(text2).casefold()
    e_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{4,}\b", t2_lower) if w not in stopwords]
    if not e_words:
        return False

    # Strip epistemic caveats & comparative contrasts from answer text before polarity matching
    t1_clean = re.sub(
        r"[^.!?\n]*\b(?:not\s+explicitly\s+(?:documented|mentioned|stated|specified|reported|listed)|не\s+зазначено\s+прямо)[^.!?\n]*",
        " ",
        strip_markdown(text1).casefold(),
    )

    for w in e_words:
        ew = re.escape(w)
        # Skip if expected itself already contains negative/rejected phrasing around this term
        t2_has_neg = bool(
            re.search(r"\b(?:not|never|не|ніколи|відхилено|rejected|без|without)\s+(?:\w+\s+){0,2}" + ew, t2_lower)
            or re.search(ew + r"\s+(?:\w+\s+){0,2}(?:not|never|не|ніколи|відхилено|rejected)\b", t2_lower)
        )
        if t2_has_neg:
            continue

        direct_neg = (
            re.search(r"\b(?:not|never|rejected|відхилено)\s+(?:\w+\s+){0,1}" + ew, t1_clean)
            or re.search(ew + r"\s+(?:is|was|were|are|був|була|було|були)?\s*(?:not|never|rejected|відхилено)\b", t1_clean)
            or re.search(r"\bне\s+(?:використову\w+|застосову\w+|підтриму\w+|обрано|вибрано)\s+(?:\w+\s+){0,1}" + ew, t1_clean)
            or re.search(ew + r"\s+(?:не\s+(?:використову\w+|застосову\w+|підтриму\w+|обрано|вибрано|використан\w+)|відхилено)\b", t1_clean)
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


CLAIM_BILINGUAL_PAIRS: Dict[str, Tuple[str, ...]] = {
    "сторінок": ("page", "pages"),
    "сторінк": ("page", "pages"),
    "page": ("сторінок", "сторінці"),
    "чанк": ("chunk", "chunks"),
    "chunk": ("чанк", "чанків", "чанки"),
    "баз": ("database", "db"),
    "database": ("база", "бази"),
    "режим": ("mode", "state"),
    "mode": ("режим", "режимі"),
    "state": ("стан", "режим"),
    "аварі": ("incident", "postmortem", "failure"),
    "відмов": ("failure", "fails", "unavailable"),
    "пошук": ("search", "retrieval", "retriever"),
    "ранжуван": ("ranking",),
    "запис": ("record", "records", "entry", "entries"),
    "товар": ("product", "products", "catalog"),
    "генерац": ("generation", "generated", "generations"),
    "зображ": ("image", "images"),
    "роздільн": ("resolution",),
    "пакет": ("batch",),
    "векторн": ("vector", "vectors"),
    "лексичн": ("lexical", "bm25"),
    "пам": ("memory", "ram", "vram"),
    "памят": ("memory", "ram", "vram"),
    "відбит": ("reflection", "reflections"),
    "дерев": ("tree", "trees"),
    "будин": ("cabin",),
    "озер": ("lake",),
    "крок": ("step", "steps"),
    "звіт": ("report",),
    "технічн": ("technical",),
    "файл": ("file", "pdf", "document"),
    "місти": ("containing", "contains", "contain", "included"),
    "обсяг": ("size", "pages"),
    "велик": ("large",),
    "специфікац": ("specification", "spec"),
    "витрат": ("overhead", "cost"),
    "секунд": ("seconds", "second", "sec"),
    "хвилин": ("minutes", "minute", "min"),
    "байт": ("byte", "bytes"),
    "розмір": ("size", "dimension"),
    "помил": ("error", "errors", "failed"),
    "крах": ("crashed", "crash"),
    "відкид": ("rejected",),
    "перезапуск": ("restarted", "restart"),
    "порог": ("threshold",),
    "перевищ": ("exceeded", "exceeding", "crossed"),
    "триваліст": ("duration", "timeline", "time"),
    "duration": ("timeline", "триваліст", "тривалість", "час"),
    "інцидент": ("incident", "outage"),
    "сервіс": ("service", "services"),
    "перерв": ("outage", "downtime"),
    "середн": ("average", "mean"),
    "відношен": ("ratio",),
}


def _parse_claim_float(tok: str) -> Optional[float]:
    if re.fullmatch(r"-?\d+(?:\.\d+)?", tok or ""):
        try:
            return float(tok)
        except Exception:
            return None
    return None


def _number_supported_by_evidence(n: float, ev_numbers: List[float]) -> bool:
    if not ev_numbers:
        return False
    if any(math.isclose(n, en, rel_tol=0.02, abs_tol=0.01) for en in ev_numbers):
        return True
    if any(
        math.isclose(n, en * f, rel_tol=0.05, abs_tol=1.0)
        or math.isclose(n * f, en, rel_tol=0.05, abs_tol=1.0)
        for en in ev_numbers
        for f in (60.0, 1000.0, 1024.0, 1000.0 ** 2, 1024.0 ** 2)
    ):
        return True
    for i, en1 in enumerate(ev_numbers):
        for en2 in ev_numbers[i:]:
            if (
                math.isclose(n, abs(en1 - en2), rel_tol=0.02, abs_tol=0.5)
                or math.isclose(n, en1 + en2, rel_tol=0.02, abs_tol=0.5)
                or math.isclose(n, (en1 + en2) / 2.0, rel_tol=0.02, abs_tol=0.05)
                or math.isclose(n, en1 * en2, rel_tol=0.02, abs_tol=0.5)
                or (en2 > 0 and math.isclose(n, en1 / en2, rel_tol=0.02, abs_tol=0.05))
                or (en1 > 0 and math.isclose(n, en2 / en1, rel_tol=0.02, abs_tol=0.05))
            ):
                return True
    return False


def _normalize_for_claims(text: str) -> str:
    """Normalize whitespace, markdown, comma-thousands, and space-thousands."""
    no_commas = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text or "")
    no_spaces = re.sub(
        r"(?<!\d)(\d{1,3}(?:[ \u00a0]\d{3})+)(?!\d)",
        lambda m: m.group(1).replace(" ", "").replace("\u00a0", ""),
        no_commas,
    )
    return normalize_text(no_spaces)


def _tokenize_claim_terms(norm_text: str) -> List[str]:
    """Extract content tokens while preserving hyphen-separated sub-terms ('450-page' -> ['450', 'page'])."""
    raw = re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{2,}\b", norm_text)
    out: List[str] = []
    for tok in raw:
        parts = [p for p in tok.split("-") if len(p) >= 2]
        if len(parts) > 1:
            out.extend(parts)
        elif len(tok) >= 3 or tok.isdigit():
            out.append(tok)
    return out


def _expand_token_set(tokens: List[str]) -> Set[str]:
    """Expand token list with morphological stems and bilingual EN<->UK technical synonyms."""
    expanded: Set[str] = set()
    for t in tokens:
        expanded.add(t)
        st = stem_word(t)
        expanded.add(st)
        if len(t) >= 5:
            expanded.add(t[:5])
        for d in (BILINGUAL_SYNONYMS, CLAIM_BILINGUAL_PAIRS):
            for key, syns in d.items():
                if t == key or st == key or (len(key) >= 4 and (t.startswith(key) or st.startswith(key))):
                    for s in syns:
                        s_cf = s.casefold()
                        expanded.add(s_cf)
                        expanded.add(stem_word(s_cf))
    return expanded


def _token_matches_expanded(ct: str, ev_raw_set: Set[str], ev_expanded_set: Set[str]) -> bool:
    if ct in ev_raw_set or ct in ev_expanded_set:
        return True
    ct_stem = stem_word(ct)
    if ct_stem in ev_expanded_set:
        return True
    if len(ct) >= 5 and any(et.startswith(ct[:5]) or ct.startswith(et[:5]) for et in ev_raw_set if len(et) >= 5):
        return True
    if len(ct) == 4 and ct in ev_raw_set:
        return True
    for d in (BILINGUAL_SYNONYMS, CLAIM_BILINGUAL_PAIRS):
        for key, syns in d.items():
            if ct == key or ct_stem == key or (len(key) >= 4 and (ct.startswith(key) or ct_stem.startswith(key))):
                for s in syns:
                    s_cf = s.casefold()
                    if s_cf in ev_expanded_set or stem_word(s_cf) in ev_expanded_set:
                        return True
    return False


def _preprocess_claim_lines(answer: str) -> List[str]:
    """Normalize grouped citations, filter non-claim structural markdown/code blocks, and merge tiny trailing fragments."""
    ans = normalize_multi_citations(answer or "")
    # Strip fenced code blocks completely so JSON/curl/bash snippet lines aren't scored as independent English prose claims
    ans = re.sub(r"```[\s\S]*?```", " ", ans)
    # Merge standalone citation-only lines onto preceding line
    ans = re.sub(
        r"\n+\s*((?:\*{0,2}\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]\*{0,2}[.,;: ]*)+)(?=\n|$)",
        r" \1",
        ans,
    )
    lines = ans.splitlines()
    filtered_lines: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            if filtered_lines:
                filtered_lines.append("")
            continue
        # Skip horizontal rules, markdown table dividers, and pure markdown headings (# Title)
        if re.match(r"^[-=*_]{3,}$", s) or re.match(r"^\|[\s:|-]+\|$", s) or re.match(r"^#{1,6}\s+", s):
            continue
        # Skip short standalone colon header labels (e.g. '**У звіті про аварію:**', 'Key findings:', 'Evidence:')
        no_cite_s = re.sub(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", "", s).strip()
        plain_s = strip_markdown(no_cite_s).strip()
        if plain_s.endswith(":") and len(plain_s.split()) <= 8:
            continue
        # Merge short non-sentence fragments (e.g. '**5 секунд**.') onto preceding line
        if filtered_lines and filtered_lines[-1] and len(plain_s.split()) <= 3 and plain_s.endswith((".", "!")):
            filtered_lines[-1] = f"{filtered_lines[-1]} {s}"
            continue
        filtered_lines.append(s)

    joined = "\n".join(filtered_lines)
    # Split on sentence punctuation followed by capital/bracket/digit+unit OR newlines
    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ\[*0-9])|\n+", joined) if s.strip()]
    return raw_sentences


def verify_claim_evidence_support(
    answer: str,
    evidence: List[EvidenceChunk],
    return_judgments: bool = False,
) -> Any:
    """Verify that substantive claims in answer are anchored in retrieved evidence chunks.
    
    Features:
    - Structural markdown / code-fence / label filtering
    - Exact [doc#chunk] provenance + bilingual EN<->UK entailment
    - Technology premise consistency check
    - Topical chunk-scoped polarity & verbatim error/rejection checking
    - Explicit verdict taxonomy: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, UNSUPPORTED_CITATION, NUMERICAL_MISMATCH
    """
    if not evidence or not answer.strip():
        return (False, []) if return_judgments else False

    chunk_norms = [_normalize_for_claims(c.chunk.text) for c in evidence]
    chunk_token_sets = []
    for cn in chunk_norms:
        c_toks = _tokenize_claim_terms(cn)
        chunk_token_sets.append((set(c_toks), _expand_token_set(c_toks)))

    combined_ev_norm = " ".join(chunk_norms)
    all_ev_lower = " ".join(c.chunk.text for c in evidence).casefold()
    combined_ev_raw_tokens = set(_tokenize_claim_terms(combined_ev_norm))
    combined_ev_expanded = _expand_token_set(list(combined_ev_raw_tokens))
    ev_numbers = _parse_numbers(combined_ev_norm)
    for c in evidence:
        for h_s, m_s in re.findall(r"\b(\d{1,2}):(\d{2})\b", c.chunk.text):
            ev_numbers.append(float(int(h_s) * 60 + int(m_s)))

    raw_sentences = _preprocess_claim_lines(answer)

    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was", "were",
        "has", "have", "had", "not", "but", "what", "which", "who", "when", "where",
        "will", "can", "may", "into", "over", "under", "each", "both", "also", "only",
        "було", "для", "або", "при", "який", "яка", "яке", "які", "так", "ні", "був",
        "according", "stated", "here", "shown", "there", "based", "document", "documents",
        "indexed", "provided", "available", "however", "whereas", "while", "note",
        "наприклад", "зокрема", "також", "згідно", "проте", "однак", "наступним", "чином",
        "calculated", "calculating", "difference", "remaining", "minus", "plus", "sum",
        "total", "обчислено", "розраховано", "різниця", "залишок", "разом", "становить",
        "стосувалося", "обсягом", "який", "якого", "таким", "чином", "великого",
    }

    judgments = []

    for raw_s in raw_sentences:
        clean_s = re.sub(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", "", raw_s).strip()
        plain_clean = strip_markdown(clean_s).strip()
        if not plain_clean or len(plain_clean) < 12 or plain_clean.endswith("?"):
            continue
        # Skip conversational offer / caveat sign-offs without empirical assertions
        if re.search(r"^(?:if you need|would you like|let me know|i recommend checking|якщо вам потрібн|якщо це стосується)", plain_clean, re.IGNORECASE):
            continue

        s_norm = _normalize_for_claims(clean_s)

        # 0. Specific citation entailment & technology premise check
        cited_tags = re.findall(r"\[([a-zA-Z0-9_.-]+)#([a-zA-Z0-9_.-]+)\]", raw_s)
        matched_cited_chunks: List[EvidenceChunk] = []
        calc_prov_m = re.match(
            r"^\s*\[Розраховано:\s*([^\]=]+)=\s*(-?\d+(?:\.\d+)?)\s*\]",
            clean_s,
            re.IGNORECASE,
        )
        calc_prov_verified = False
        if cited_tags:
            specific_chunk_supported = False
            tech_premise_violation = None
            for doc_ref, chunk_ref in cited_tags:
                d_ref_cf = doc_ref.casefold()
                matched_chunks = [
                    c for c in evidence
                    if (c.chunk.chunk_id == chunk_ref or c.chunk.chunk_id.endswith(chunk_ref))
                    and (
                        c.chunk.document_id.casefold() == d_ref_cf
                        or Path(c.chunk.document_id).stem.casefold() == d_ref_cf
                        or str(c.chunk.metadata.get("source_id", "")).casefold() == d_ref_cf
                        or Path(str(c.chunk.metadata.get("filename", ""))).stem.casefold() == d_ref_cf
                    )
                ]
                matched_cited_chunks.extend(matched_chunks)
                for mc in matched_chunks:
                    mc_lower = mc.chunk.text.casefold()
                    absent_techs = [
                        tech for tech in KNOWN_TECHNOLOGIES
                        if re.search(r"\b" + re.escape(tech) + r"\b", s_norm)
                        and not re.search(r"\b" + re.escape(tech) + r"\b", mc_lower)
                    ]
                    if absent_techs:
                        tech_premise_violation = absent_techs[0]
                        continue
                    mc_norm = _normalize_for_claims(mc.chunk.text)
                    mc_toks = _tokenize_claim_terms(mc_norm)
                    mc_raw_set = set(mc_toks)
                    mc_exp_set = _expand_token_set(mc_toks)
                    claim_tokens = [w for w in _tokenize_claim_terms(s_norm) if w not in stopwords]
                    if claim_tokens:
                        c_overlap = sum(
                            1 for ct in claim_tokens
                            if _token_matches_expanded(ct, mc_raw_set, mc_exp_set)
                        ) / len(claim_tokens)
                        if c_overlap >= 0.25:
                            specific_chunk_supported = True
                            break
                if specific_chunk_supported:
                    break
            if not specific_chunk_supported and calc_prov_m and matched_cited_chunks:
                lhs_raw = calc_prov_m.group(1).replace("×", "*").strip()
                rhs_val = float(calc_prov_m.group(2))
                try:
                    eval_res = safe_calculate(lhs_raw)["result"]
                except Exception:
                    eval_res = None
                lhs_nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", lhs_raw)]
                mc_all_nums: List[float] = []
                for mc in matched_cited_chunks:
                    mc_all_nums.extend(_parse_numbers(_normalize_for_claims(mc.chunk.text)))
                    for h_s, m_s in re.findall(r"\b(\d{1,2}):(\d{2})\b", mc.chunk.text):
                        mc_all_nums.append(float(int(h_s) * 60 + int(m_s)))
                unit_constants = {2.0, 60.0, 100.0, 1000.0, 1024.0, 3600.0}
                any_from_mc = any(
                    any(math.isclose(n, evn, rel_tol=1e-4, abs_tol=1e-3) for evn in mc_all_nums)
                    for n in lhs_nums
                )
                all_lhs_ok = bool(lhs_nums) and all(
                    any(math.isclose(n, evn, rel_tol=1e-4, abs_tol=1e-3) for evn in mc_all_nums)
                    or n in unit_constants
                    for n in lhs_nums
                )
                if (
                    eval_res is not None
                    and math.isclose(eval_res, rhs_val, rel_tol=1e-2, abs_tol=0.05)
                    and any_from_mc
                    and all_lhs_ok
                    and not tech_premise_violation
                ):
                    specific_chunk_supported = True
                    calc_prov_verified = True
            if calc_prov_verified:
                judgments.append({
                    "claim": clean_s,
                    "supported": True,
                    "verdict": "SUPPORTED",
                    "support_score": 1.0,
                })
                continue
            if not specific_chunk_supported:
                reason = (
                    f"Subject '{tech_premise_violation}' absent from cited chunk(s) {cited_tags}."
                    if tech_premise_violation
                    else f"Cited chunk(s) {cited_tags} do not entail the claim sentence."
                )
                judgments.append({
                    "claim": clean_s,
                    "supported": False,
                    "verdict": "UNSUPPORTED_CITATION",
                    "reason": reason,
                })
                continue

        # 0b. Technology premise check even on uncited claims
        absent_global_techs = [
            tech for tech in KNOWN_TECHNOLOGIES
            if re.search(r"\b" + re.escape(tech) + r"\b", s_norm)
            and not re.search(r"\b" + re.escape(tech) + r"\b", all_ev_lower)
        ]
        if absent_global_techs and not re.search(r"\b(?:not|never|не|ніколи|відсутн|немає|ані)\b", s_norm):
            judgments.append({
                "claim": clean_s,
                "supported": False,
                "verdict": "INSUFFICIENT_EVIDENCE",
                "reason": f"Subject '{absent_global_techs[0]}' absent from retrieved evidence.",
            })
            continue

        # 1. Contradiction & negation detection against topical evidence
        is_epistemic_caveat = bool(re.search(
            r"\b(?:not\s+explicitly\s+(?:documented|mentioned|stated|reported|listed|configured)|не\s+зазначено\s+прямо)\b",
            s_norm,
        ))
        has_neg = bool(re.search(r"\b(?:not|never|neither|nor|rejected|failed|cannot|isn't|wasn't|doesn't|didn't|відхилено)\b", s_norm))
        if has_neg and not is_epistemic_caveat:
            core_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\u0400-\u04FF.-]{4,}\b", s_norm) if w not in stopwords]
            is_negated = any(
                re.search(r"\b(?:not|never|rejected|відхилено)\s+(?:\w+\s+){0,1}" + re.escape(w), s_norm)
                or re.search(re.escape(w) + r"\s+(?:is|was|were|are|був|була|було|були)?\s*(?:not|never|rejected|відхилено)\b", s_norm)
                for w in core_words
            )
            if is_negated and core_words:
                # Select topical chunks matching this claim
                candidate_indices = (
                    [i for i, ec in enumerate(evidence) if ec in matched_cited_chunks]
                    if matched_cited_chunks
                    else list(range(len(evidence)))
                )
                topical_indices = []
                for idx in candidate_indices:
                    ev_raw_s, ev_exp_s = chunk_token_sets[idx]
                    hits = sum(1 for w in core_words if _token_matches_expanded(w, ev_raw_s, ev_exp_s))
                    min_hits = 2 if len(core_words) >= 3 else 1
                    if hits >= min_hits:
                        topical_indices.append(idx)

                if topical_indices:
                    ev_already_has_neg = any(
                        any(
                            re.search(r"\b(?:not|never|rejected|failed|unable|without|не|відхилено|жоден)\s+(?:\w+\s+){0,2}" + re.escape(w), chunk_norms[i])
                            or re.search(re.escape(w) + r"\s+(?:\w+\s+){0,2}(?:not|never|rejected|failed|unable|не|відхилено)\b", chunk_norms[i])
                            for w in core_words
                        )
                        or any(phrase in chunk_norms[i] for phrase in ("not allowed", "rejected", "crashed", "заблокований", "не проводилося", "не застосовувалися"))
                        for i in topical_indices
                    )
                    if not ev_already_has_neg:
                        judgments.append({
                            "claim": clean_s,
                            "supported": False,
                            "verdict": "CONTRADICTED",
                            "reason": "Claim asserts negative/rejection while topical evidence asserts affirmative.",
                        })
                        continue

        # 2. Numerical verification with derived numbers & time/unit scale allowance
        claim_nums = _parse_numbers(re.sub(r"https?://\S+", " ", s_norm))
        substantive_nums = [
            n for n in claim_nums
            if n not in {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1234.0, 6380.0, 8000.0, 8501.0, 200.0, 400.0, 500.0, 503.0, 2024.0, 2025.0, 2026.0}
        ]
        if substantive_nums and ev_numbers:
            unsupported_nums = [
                n for n in substantive_nums
                if not _number_supported_by_evidence(n, ev_numbers)
            ]
            if len(unsupported_nums) > 2:
                judgments.append({
                    "claim": clean_s,
                    "supported": False,
                    "verdict": "NUMERICAL_MISMATCH",
                    "reason": f"Numbers {unsupported_nums} absent from evidence.",
                })
                continue

        # 3. Grounding in cumulative evidence chunks (with bilingual stem/synonym + derived-number alignment)
        claim_tokens = [w for w in _tokenize_claim_terms(s_norm) if w not in stopwords]
        if not claim_tokens:
            continue

        non_num_matched = sum(
            1 for ct in claim_tokens
            if _parse_claim_float(ct) is None
            and _token_matches_expanded(ct, combined_ev_raw_tokens, combined_ev_expanded)
        )
        num_matched = 0
        for ct in claim_tokens:
            num_val = _parse_claim_float(ct)
            if num_val is None:
                continue
            if _token_matches_expanded(ct, combined_ev_raw_tokens, combined_ev_expanded) or (
                non_num_matched > 0 and ev_numbers and _number_supported_by_evidence(num_val, ev_numbers)
            ):
                num_matched += 1
        matched_cnt = non_num_matched + num_matched
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
                "verdict": "INSUFFICIENT_EVIDENCE",
                "reason": f"Only {coverage:.2f} content coverage against evidence.",
            })

    if not judgments:
        is_supported = True
        return (is_supported, []) if return_judgments else is_supported

    supported_count = sum(1 for j in judgments if j["supported"])
    support_rate = supported_count / len(judgments)
    is_supported = (support_rate >= 0.70) and not any(
        j.get("verdict") in ("CONTRADICTED", "UNSUPPORTED_CITATION", "NUMERICAL_MISMATCH")
        for j in judgments
    )

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
    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6380")
    redis_adapter = RedisVectorAdapter(
        redis_url=redis_url,
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
            refused = None
            cit_rate = None
            fab_list = []
            fact_correct = None
            claim_supp = None
            claim_judgments = []
            num_correct = None

            if is_refusal_req:
                refused = is_refusal_answer(turn.canonical_answer, turn.grounding_status)
                refusal_evals.append(1 if refused else 0)
            else:
                # 2. Citation validity evaluation (exact (doc, chunk) matching & penalty for fabrications)
                if not turn.citations:
                    # Missing citations on supported queries receive 0.0 (NO FREE PASS)
                    cit_rate = 0.0
                    citation_evals.append(0.0)
                else:
                    audit = audit_citations(turn.citations, turn.evidence)
                    fab_list = audit["fabricated_citations"]
                    if fab_list:
                        total_fabricated_citations += len(fab_list)
                    cit_rate = audit["validity_rate"]
                    citation_evals.append(cit_rate)

                # 3. Factual answer correctness
                fact_correct = compute_factual_correctness(
                    answer=turn.canonical_answer,
                    expected=item.get("expected_answer", ""),
                    evidence_spans=item.get("evidence_spans", []),
                )
                factual_evals.append(1 if fact_correct else 0)

                # 4. Claim-to-evidence support (with explicit per-claim verdicts)
                claim_supp, claim_judgments = verify_claim_evidence_support(
                    turn.canonical_answer, turn.evidence, return_judgments=True
                )
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
                "refusal_expected": is_refusal_req,
                "answer": turn.canonical_answer,
                "grounding_status": turn.grounding_status.value,
                "tools_executed": tools_called,
                "tool_executions": [rec.to_dict() for rec in turn.tool_executions],
                "citations": turn.citations,
                "fabricated_citations": fab_list,
                "citation_validity_rate": cit_rate,
                "factual_correct": fact_correct,
                "claim_supported": claim_supp,
                "claim_verdicts": [j.get("verdict") for j in claim_judgments],
                "refusal_detected": refused,
                "numerical_match": num_correct,
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
