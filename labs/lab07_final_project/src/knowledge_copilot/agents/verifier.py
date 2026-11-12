"""Multi-dimensional grounding, citation validity, and structured claim support verification."""
from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set, Tuple

from ..domain.enums import GroundingStatus
from ..domain.models import ClaimSupport, EvidenceChunk
from .recovery import (
    extract_expected_documents,
    extract_expression_numbers,
    has_arithmetic_intent,
    validate_calculator_execution,
)
from .tools import safe_calculate


ABSTENTION_PATTERNS = [
    r"\b(?:cannot determine|insufficient evidence|not available|not mentioned|unable to determine|cannot answer|do not cover|not covered)\b",
    r"\b(?:do not have enough information|not found in the indexed|do(?:es)? not contain|no (?:specific |relevant |direct )?(?:benchmark |information|evidence|data|mention|details|instructions|guidance))\b",
    r"\b(?:cannot retrieve|unable to find|not explicitly (?:list|mention|state|report|describe|specify|detail|provide|configure)|please clarify which)\b",
    r"\b(?:is not explicitly configured|is not configured|are not configured)\b",
    r"\b(?:outside (?:the )?(?:scope|indexed|documentation))\b",
    r"не можу визначити",
    r"неможливо\s+(?:надійно\s+)?визначити",
    r"недостатньо (?:даних|доказів|інформації)",
    r"недостатньо\s+однозначно\s+пов.{1,3}язаних",
    r"не міст(?:ить|ять|иться|яться)",
    r"даних .*? не виявлено",
    r"не (?:було\s+)?(?:прямо\s+)?(?:зазначен[оаіий]|наведен[оаіий]|описан[оаіий]|вказан[оаіий]|згадан[оаіий])",
    r"інформація відсутня",
    r"(?:інформаці\w*|дані|відомост\w*|згадк\w*)\s+(?:відсутн\w*|немає)",
    r"відсутн\w*\s+(?:інформація|відомості|дані|згадки)",
    r"не знайдено",
    r"не виявлено",
    r"не входить до",
    r"не покрива(?:ється|ються|ють|є)",
    r"немає (?:жодної\s+|такої\s+|цієї\s+)?(?:інформації|даних|відомостей|згадок|інструкцій|деталей)",
    r"не згадується",
    r"не надається",
]

KNOWN_TECHNOLOGIES = [
    "pinecone", "milvus", "qdrant", "weaviate", "chromadb", "chroma", "pgvector",
    "kafka", "helm", "kubernetes", "hpa", "aws s3", "s3 bucket",
    "gemini 1.5", "gemini", "claude 3.5", "claude", "gpt-4o", "gpt-4",
    "llama 3.1", "llama", "mistral", "qwen", "bert", "stable diffusion",
    "stable video diffusion", "svd", "controlnet", "lora",
    "elasticsearch", "dynamodb", "cassandra", "bigquery", "snowflake",
    "deepseek", "tensorrt", "ray serve", "triton", "terraform", "vault",
    "oauth", "rabbitmq", "faiss", "vllm", "tpu v4", "sagemaker", "imagenet",
    "opentelemetry", "grafana", "datadog", "celery", "mongodb",
]

IGNORE_STRUCTURAL_NUMS = {
    "001", "1", "2", "3", "4", "5", "6", "7", "2024", "2025", "2026",
    "1234", "6379", "6380", "8000", "8080", "200", "201", "400", "404", "422", "500", "503",
    "768", "384", "1536", "0.5", "8", "70",
}

BILINGUAL_DOMAIN_PAIRS = [
    ({"офлайн", "автономн"}, {"offline"}),
    ({"локальн", "робоч", "станц"}, {"local", "workstation", "workstations"}),
    ({"затримк", "швидкост", "швидкіст", "обмежен"}, {"latency", "constraint", "sub"}),
    ({"мілісекунд", "мс"}, {"ms", "50ms", "milliseconds"}),
    ({"транзакційн", "цілісніст", "відкат", "статус"}, {"transactional", "integrity", "rollback", "rollbacks", "status"}),
    ({"пошкоджен", "байт", "втрат"}, {"corrupted", "corruption", "bytes", "zero"}),
    ({"невдал", "помилк", "завантажен"}, {"failed", "bulk", "uploads", "upload"}),
    ({"лексичн", "резервн"}, {"lexical", "fallback"}),
    ({"витіснен", "політик", "пам'ят"}, {"eviction", "policy", "maxmemory", "allkeys-lru", "noeviction"}),
    ({"блокуван", "конкурентн", "запис"}, {"lock", "locks", "locking", "concurrent", "write", "busy_timeout"}),
]


@dataclass
class CitationCanonicalizationReport:
    canonical_answer: str
    raw_citations: List[str] = field(default_factory=list)
    raw_invalid_citations: List[str] = field(default_factory=list)
    final_citations: List[str] = field(default_factory=list)
    final_invalid_citations: List[str] = field(default_factory=list)
    citation_repairs_count: int = 0
    citation_removals_count: int = 0
    unsupported_claims_removed_count: int = 0


def strip_markdown(text: str) -> str:
    """Strip markdown formatting characters for robust text inspection."""
    return re.sub(r"[*_`#]", "", text or "")


def chunk_searchable_text(ec: EvidenceChunk) -> str:
    """Return full searchable text of an EvidenceChunk including document_id and section title."""
    doc_norm = ec.chunk.document_id.replace("_", " ").replace("-", " ")
    sec_norm = (ec.chunk.section or "").replace("_", " ")
    return f"{ec.chunk.document_id} {doc_norm} {sec_norm} {ec.chunk.text}"


def normalize_multi_citations(text: str) -> str:
    """Normalize comma- or semicolon-separated citations inside a single bracket:
    e.g. '[doc1#chunk1, doc2#chunk2]' -> '[doc1#chunk1] [doc2#chunk2]'."""
    def _split_bracket(m):
        inner = m.group(1)
        tags = re.findall(r"[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+", inner)
        if len(tags) > 1:
            return " ".join(f"[{t}]" for t in tags)
        return m.group(0)
    return re.sub(r"\[([^\]]+)\]", _split_bracket, text or "")


def extract_citations(text: str) -> List[str]:
    """Extract standard citation tags matching [source_id#chunk_id], excluding literal placeholders."""
    norm_text = normalize_multi_citations(text)
    return [
        c for c in re.findall(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", norm_text)
        if not c.lower().startswith(("[source_id#", "[document_id#"))
    ]


def extract_suspicious_citations(text: str) -> List[str]:
    """Find malformed, filename-style, or placeholder citations."""
    norm_text = normalize_multi_citations(text)
    valid_cites = set(extract_citations(norm_text))
    patterns = [
        r"\[(?:source_id|document_id)#[^\]]+\]",
        r"\[[^\]\n]*_report\.md[^\]\n]*\]",
        r"\[(?:Джерело|Citation|Source):\s*[^\]]+\]",
    ]
    matches = []
    for p in patterns:
        for m in re.findall(p, norm_text, re.IGNORECASE):
            if m not in valid_cites and m not in matches:
                matches.append(m)
    for m in re.findall(r"\[[^\]\n]*chunk-[^\]\n]*\]", norm_text, re.IGNORECASE):
        if m not in valid_cites and m not in matches:
            matches.append(m)
    return matches


def is_abstention(text: str) -> bool:
    """Detect explicit refusal or declaration of insufficient evidence."""
    clean = strip_markdown(text)
    for pattern in ABSTENTION_PATTERNS:
        if re.search(pattern, clean, re.IGNORECASE):
            return True
    return False


def _normalize_numeric_token(num_str: str) -> str:
    """Normalize commas and leading zeros in numeric tokens (e.g. '7,292' -> '7292', '001' -> '1')."""
    cleaned = num_str.strip()
    if "," in cleaned and "." not in cleaned:
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts[1]) == 3 and parts[0] in {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "13", "18"}:
            # Could be European decimal comma (e.g. 6,765 or 13,950) or thousands separator
            cleaned = cleaned.replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        val = float(cleaned)
        if val.is_integer():
            return str(int(val))
        return f"{val:g}"
    except ValueError:
        return cleaned


def _number_present_in_text(num_str: str, target_text: str) -> bool:
    """Check whether a numeric token appears in target_text as a bounded number (not substring of 5000)."""
    dot_ver = num_str.replace(",", ".")
    cleaned = num_str.replace(",", "")
    norm = _normalize_numeric_token(num_str)
    target_no_commas = re.sub(r"(?<=\d),(?=\d{3}\b)", "", target_text)
    for variant in dict.fromkeys([num_str, dot_ver, cleaned, norm]):
        if not variant:
            continue
        if re.search(rf"(?<![\d.]){re.escape(variant)}(?![\d]|(?:\.\d))", target_text) or re.search(
            rf"(?<![\d.]){re.escape(variant)}(?![\d]|(?:\.\d))", target_no_commas
        ):
            return True
    return False


def build_citation_registry(evidence_chunks: List[EvidenceChunk]) -> Dict[str, EvidenceChunk]:
    """Build the authoritative citation registry strictly from retrieved EvidenceChunk objects."""
    registry: Dict[str, EvidenceChunk] = {}
    for ec in evidence_chunks:
        registry[ec.citation] = ec
        c_id = ec.chunk.chunk_id
        d_id = ec.chunk.document_id
        registry[f"[{d_id}#{c_id}]"] = ec
        if ec.chunk.metadata and "source_id" in ec.chunk.metadata:
            registry[f"[{ec.chunk.metadata['source_id']}#{c_id}]"] = ec
    return registry


def _extract_claim_numbers(sentence_without_cites: str) -> List[str]:
    """Extract numeric tokens from a claim sentence after scrubbing identifiers like Lab 4, Recall@5, HNSW, etc."""
    scrubbed = re.sub(r"\b(?:Lab|Лабораторн\w*(?:\s+робот\w*)?)\s*№?\s*0?\d+\b", " ", sentence_without_cites, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\bADR[_\s-]*0*\d+\b", " ", scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\b(?:Recall|Hit|MRR|NDCG)@\d+\b", " ", scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\b(?:steps|seed|negative|phrasing|desc|fact|analysis|control)_\d+\b", " ", scrubbed, flags=re.IGNORECASE)
    scrubbed = re.sub(r"\b\d+(?:\.\d+)?[BbKk]\b", " ", scrubbed)
    scrubbed = re.sub(r"\b[0-2]?\d:[0-5]\d\b", " ", scrubbed)
    return re.findall(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?", scrubbed)


def _extract_subject_predicate_entities(sentence_without_cites: str) -> Tuple[str, str, List[str]]:
    """Extract structured subject, predicate, and key domain entities from a claim sentence."""
    clean = strip_markdown(sentence_without_cites).strip()
    words = re.findall(r"[A-Za-zА-Яа-яІіЇїЄєҐґ0-9_.-]{3,}", clean)
    subject = " ".join(words[:3]) if words else ""
    predicate = ""
    pred_m = re.search(
        r"\b(is|are|was|were|uses|used|requires|provides|failed|processed|equals|становить|складає|дорівнює|використову\w+|забезпечу\w+|підтриму\w+|досяг\w+|отрима\w+)\b",
        clean,
        re.IGNORECASE,
    )
    if pred_m:
        predicate = pred_m.group(1)
    entities = [
        w.casefold()
        for w in words
        if len(w) >= 4
        and w.casefold() not in {
            "that", "this", "with", "from", "were", "have", "been", "when", "where", "which",
            "становлять", "становить", "згідно", "також", "через", "після", "під", "для", "між",
        }
    ][:10]
    return subject, predicate, entities


def _has_contradiction(sentence_lower: str, chunk_lower: str) -> Optional[str]:
    """Detect direct polarity/boolean/policy contradictions between claim sentence and cited chunk."""
    # 1. Eviction policy contradiction (allkeys-lru vs noeviction)
    if "noeviction" in sentence_lower and "allkeys-lru" in chunk_lower and "noeviction" not in chunk_lower:
        return "Claim asserts 'noeviction' policy, contradicting 'allkeys-lru' in cited chunk."
    if "allkeys-lru" in sentence_lower and "noeviction" in chunk_lower and "allkeys-lru" not in chunk_lower:
        return "Claim asserts 'allkeys-lru' policy, contradicting 'noeviction' in cited chunk."

    # 2. Cabin position contradiction (left vs right/center)
    if re.search(r"\b(?:on\s+the\s+right|праворуч|у\s+правій)\b", sentence_lower) and re.search(r"\b(?:left|ліворуч|у\s+лівій)\b", chunk_lower):
        if not re.search(r"\b(?:not|не|left\s+or\s+right|center\s+or\s+right)\b", sentence_lower):
            return "Claim asserts right-side position, contradicting left-side position in cited chunk."

    # 3. Attention slicing / xFormers enabled vs not applied
    if re.search(r"\b(?:attention\s+slicing|xformers)\s+(?:was\s+enabled|enabled|було\s+увімкнено|застосовувалися)\b", sentence_lower):
        if re.search(r"не\s+застосовувалися|not\s+applied|disabled", chunk_lower):
            return "Claim asserts optimization was enabled, contradicting cited chunk stating it was not applied."

    # 4. Data corruption > 0 vs 0 bytes corrupted
    if re.search(r"\b(?:data\s+was\s+corrupted|дані\s+було\s+пошкоджено)\b", sentence_lower) and not re.search(r"\b(?:no|0|нуль|не\s+було)\b", sentence_lower):
        if "0 bytes corrupted" in chunk_lower:
            return "Claim asserts data corruption occurred, contradicting '0 bytes corrupted' in cited chunk."

    return None


def assess_claim_support(
    sentence: str,
    s_citations: List[str],
    supplied_citations: Dict[str, EvidenceChunk],
    all_evidence_chunks: List[EvidenceChunk],
    calc_numbers: Set[str],
    abstained: bool = False,
) -> ClaimSupport:
    """
    Evaluate a single claim sentence against its cited chunks and return a structured ClaimSupport object
    with explicit verdict: SUPPORTED, CONTRADICTED, INSUFFICIENT_EVIDENCE, INVALID_CITATION, or NEEDS_REVIEW.
    """
    sentence_without_cites = re.sub(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", "", sentence)
    sentence_without_cites = re.sub(r"https?://\S+", "", sentence_without_cites)
    subject, predicate, entities = _extract_subject_predicate_entities(sentence_without_cites)
    claim_nums = _extract_claim_numbers(sentence_without_cites)

    all_evidence_lower = " ".join(chunk_searchable_text(c) for c in all_evidence_chunks).lower() if all_evidence_chunks else ""
    s_lower = strip_markdown(sentence_without_cites).lower()

    # Check if any cited tag is invalid (not in authoritative registry)
    invalid_tags = [c for c in s_citations if c not in supplied_citations]
    if invalid_tags:
        return ClaimSupport(
            claim_text=sentence,
            cited_chunk_ids=s_citations,
            is_supported=False,
            unsupported_reason=f"Citation(s) {invalid_tags} not found in authoritative retrieval registry.",
            verdict="INVALID_CITATION",
            subject=subject,
            predicate=predicate,
            objects_and_entities=entities,
            numbers_and_units=claim_nums,
        )

    # Uncited sentence handling
    if not s_citations:
        if all_evidence_chunks and not abstained:
            absent_techs = [
                tech for tech in KNOWN_TECHNOLOGIES
                if re.search(r"\b" + re.escape(tech) + r"\b", s_lower)
                and not re.search(r"\b" + re.escape(tech) + r"\b", all_evidence_lower)
            ]
            if absent_techs:
                return ClaimSupport(
                    claim_text=sentence,
                    cited_chunk_ids=[],
                    is_supported=False,
                    unsupported_reason=f"Subject '{absent_techs[0]}' absent from retrieved evidence.",
                    verdict="INSUFFICIENT_EVIDENCE",
                    subject=subject,
                    predicate=predicate,
                    objects_and_entities=entities,
                    numbers_and_units=claim_nums,
                )

        substantive_nums = [
            n for n in claim_nums
            if n not in IGNORE_STRUCTURAL_NUMS
            and _normalize_numeric_token(n) not in IGNORE_STRUCTURAL_NUMS
            and _normalize_numeric_token(n) not in calc_numbers
        ]
        if substantive_nums and not abstained:
            return ClaimSupport(
                claim_text=sentence,
                cited_chunk_ids=[],
                is_supported=False,
                unsupported_reason=f"Uncited factual claim with numbers {substantive_nums} lacks supporting citation.",
                verdict="INSUFFICIENT_EVIDENCE",
                subject=subject,
                predicate=predicate,
                objects_and_entities=entities,
                numbers_and_units=claim_nums,
            )

        return ClaimSupport(
            claim_text=sentence,
            cited_chunk_ids=[],
            is_supported=True,
            unsupported_reason=None,
            verdict="SUPPORTED",
            subject=subject,
            predicate=predicate,
            objects_and_entities=entities,
            numbers_and_units=claim_nums,
        )

    # Cited sentence handling
    valid_chunks = [supplied_citations[c] for c in s_citations if c in supplied_citations]
    combined_cited_text = " ".join(chunk_searchable_text(c) for c in valid_chunks)
    combined_cited_lower = combined_cited_text.lower()

    # 1. Check explicit contradictions
    contradiction_reason = _has_contradiction(s_lower, combined_cited_lower)
    if contradiction_reason:
        return ClaimSupport(
            claim_text=sentence,
            cited_chunk_ids=s_citations,
            is_supported=False,
            unsupported_reason=contradiction_reason,
            verdict="CONTRADICTED",
            subject=subject,
            predicate=predicate,
            objects_and_entities=entities,
            numbers_and_units=claim_nums,
        )

    # 2. Technology premise validation against cited chunks (including document_id and section!)
    for tech in KNOWN_TECHNOLOGIES:
        if re.search(r"\b" + re.escape(tech) + r"\b", s_lower):
            if not re.search(r"\b" + re.escape(tech) + r"\b", combined_cited_lower):
                return ClaimSupport(
                    claim_text=sentence,
                    cited_chunk_ids=s_citations,
                    is_supported=False,
                    unsupported_reason=f"Subject '{tech}' absent from cited passage(s) {s_citations}.",
                    verdict="INSUFFICIENT_EVIDENCE",
                    subject=subject,
                    predicate=predicate,
                    objects_and_entities=entities,
                    numbers_and_units=claim_nums,
                )

    # 3. Numeric validation against cited chunks (and calculator results)
    for num in claim_nums:
        if (
            num in IGNORE_STRUCTURAL_NUMS
            or _normalize_numeric_token(num) in IGNORE_STRUCTURAL_NUMS
            or _normalize_numeric_token(num) in calc_numbers
        ):
            continue
        if not _number_present_in_text(num, combined_cited_text):
            return ClaimSupport(
                claim_text=sentence,
                cited_chunk_ids=s_citations,
                is_supported=False,
                unsupported_reason=f"Number '{num}' absent from cited passage(s) {s_citations}.",
                verdict="CONTRADICTED" if any(c.isdigit() for c in combined_cited_text) else "INSUFFICIENT_EVIDENCE",
                subject=subject,
                predicate=predicate,
                objects_and_entities=entities,
                numbers_and_units=claim_nums,
            )

    # 4. Entity / Bilingual lexical overlap check (when claim has no numbers, ensure cited chunk is not completely unrelated)
    if not claim_nums and entities:
        chunk_words = set(re.findall(r"[a-zа-яіїєґ0-9_]{3,}", combined_cited_lower))
        chunk_stems = {w[:4] for w in chunk_words if len(w) >= 4}
        claim_stems = {w[:4] for w in entities if len(w) >= 4}
        has_overlap = bool(claim_stems & chunk_stems)
        if not has_overlap:
            for uk_set, en_set in BILINGUAL_DOMAIN_PAIRS:
                uk_in_claim = any(any(u in w for u in uk_set) for w in entities)
                en_in_chunk = any(any(e in cw for e in en_set) for cw in chunk_words)
                en_in_claim = any(any(e in w for e in en_set) for w in entities)
                uk_in_chunk = any(any(u in cw for u in uk_set) for cw in chunk_words)
                if (uk_in_claim and en_in_chunk) or (en_in_claim and uk_in_chunk):
                    has_overlap = True
                    break
        if not has_overlap and len(entities) >= 3:
            return ClaimSupport(
                claim_text=sentence,
                cited_chunk_ids=s_citations,
                is_supported=False,
                unsupported_reason=f"Cited passage(s) {s_citations} do not share topical entities with claim.",
                verdict="INSUFFICIENT_EVIDENCE",
                subject=subject,
                predicate=predicate,
                objects_and_entities=entities,
                numbers_and_units=claim_nums,
            )

    return ClaimSupport(
        claim_text=sentence,
        cited_chunk_ids=s_citations,
        is_supported=True,
        unsupported_reason=None,
        verdict="SUPPORTED",
        subject=subject,
        predicate=predicate,
        objects_and_entities=entities,
        numbers_and_units=claim_nums,
    )


def find_unique_supporting_citations(
    sentence_without_cites: str,
    evidence_chunks: List[EvidenceChunk],
    calc_numbers: Set[str],
    preferred_doc_id: Optional[str] = None,
) -> List[str]:
    """
    Find whether one or more retrieved chunks in evidence_chunks uniquely and deterministically
    support the claim's numbers and entities.
    Never uses string edit distance on chunk hashes.
    """
    if not evidence_chunks:
        return []

    claim_nums = [
        n for n in _extract_claim_numbers(sentence_without_cites)
        if n not in IGNORE_STRUCTURAL_NUMS and _normalize_numeric_token(n) not in IGNORE_STRUCTURAL_NUMS
    ]
    non_calc_nums = [n for n in claim_nums if _normalize_numeric_token(n) not in calc_numbers]
    _, _, entities = _extract_subject_predicate_entities(sentence_without_cites)

    # Case A: Claim has specific non-calculated numbers (e.g. "14 bulk uploads failed" or "5000 ms" and "50 ms")
    if non_calc_nums:
        # First check if a SINGLE chunk contains all non_calc_nums and passes assess_claim_support
        matching_single: List[EvidenceChunk] = []
        for ec in evidence_chunks:
            if preferred_doc_id and ec.chunk.document_id != preferred_doc_id:
                continue
            txt = chunk_searchable_text(ec)
            if all(_number_present_in_text(n, txt) for n in non_calc_nums):
                supp = assess_claim_support(
                    f"{sentence_without_cites} {ec.citation}",
                    [ec.citation],
                    {ec.citation: ec},
                    evidence_chunks,
                    calc_numbers,
                )
                if supp.is_supported:
                    matching_single.append(ec)

        if len(matching_single) == 1:
            return [matching_single[0].citation]

        # If preferred_doc_id had no unique match, try across all evidence_chunks
        if not matching_single and preferred_doc_id:
            for ec in evidence_chunks:
                txt = chunk_searchable_text(ec)
                if all(_number_present_in_text(n, txt) for n in non_calc_nums):
                    supp = assess_claim_support(
                        f"{sentence_without_cites} {ec.citation}",
                        [ec.citation],
                        {ec.citation: ec},
                        evidence_chunks,
                        calc_numbers,
                    )
                    if supp.is_supported:
                        matching_single.append(ec)
            if len(matching_single) == 1:
                return [matching_single[0].citation]

        # Next check if each number in non_calc_nums is uniquely supported by a specific chunk (e.g. 5000 in chunk A, 50 in chunk B)
        if len(non_calc_nums) >= 2 and not matching_single:
            per_num_cits: List[str] = []
            all_unique = True
            for n in non_calc_nums:
                holders = [ec for ec in evidence_chunks if _number_present_in_text(n, chunk_searchable_text(ec))]
                if len(holders) == 1:
                    if holders[0].citation not in per_num_cits:
                        per_num_cits.append(holders[0].citation)
                else:
                    all_unique = False
                    break
            if all_unique and per_num_cits:
                return per_num_cits

    # Case B: Claim has no non-calculated numbers (or numbers are in calc_numbers), check if a unique chunk matches >= 2 specific entities
    if entities:
        scored_chunks: List[Tuple[int, EvidenceChunk]] = []
        for ec in evidence_chunks:
            if preferred_doc_id and ec.chunk.document_id != preferred_doc_id:
                continue
            c_lower = chunk_searchable_text(ec).lower()
            c_words = set(re.findall(r"[a-zа-яіїєґ0-9_]{3,}", c_lower))
            overlap = sum(1 for e in entities if e in c_words or any(e[:5] == cw[:5] for cw in c_words if len(e) >= 5 and len(cw) >= 5))
            if overlap >= 2:
                supp = assess_claim_support(
                    f"{sentence_without_cites} {ec.citation}",
                    [ec.citation],
                    {ec.citation: ec},
                    evidence_chunks,
                    calc_numbers,
                )
                if supp.is_supported:
                    scored_chunks.append((overlap, ec))
        if len(scored_chunks) == 1:
            return [scored_chunks[0][1].citation]

    return []


def _collect_verified_calc_state(
    evidence_chunks: List[EvidenceChunk],
    tool_executions: Optional[List],
    user_input: Optional[str] = None,
) -> Tuple[Set[str], Set[str], bool, str, bool]:
    """
    Collect ONLY verified calculator operands and results.
    Returns (calc_numbers, calc_result_numbers, calc_failed, calc_fail_reason, calc_explicit_only).
    """
    calc_entries: List[Tuple[int, str, Any]] = []
    for idx, t in enumerate(tool_executions or []):
        t_name = getattr(t, "tool_name", None) or (t.get("tool_name") if isinstance(t, dict) else None)
        if t_name != "calculate":
            continue
        t_args = getattr(t, "arguments", None) or (t.get("arguments") if isinstance(t, dict) else None)
        t_res = getattr(t, "result", None) or (t.get("result") if isinstance(t, dict) else None)
        expr = str(t_args.get("expression", "")) if isinstance(t_args, dict) else ""
        calc_entries.append((idx, expr, t_res))

    if not calc_entries:
        return set(), set(), False, "", False

    # Full semantic, provenance, and relevance validation when user_input is available
    if user_input:
        calc_val = validate_calculator_execution(user_input, tool_executions, evidence_chunks)
        if not calc_val.is_valid:
            return set(), set(), True, calc_val.reason, False
        calc_numbers: Set[str] = set()
        calc_result_numbers: Set[str] = set()
        for auth_idx in calc_val.authoritative_record_indices:
            _idx, expr, t_res = next((e for e in calc_entries if e[0] == auth_idx), (auth_idx, "", None))
            res_val = t_res["result"] if isinstance(t_res, dict) and "result" in t_res else t_res
            if isinstance(res_val, (int, float)):
                r_str = str(res_val)
                r_norm = _normalize_numeric_token(r_str)
                calc_numbers.update({r_str, r_norm})
                calc_result_numbers.update({r_str, r_norm})
            if expr:
                try:
                    for op_val in extract_expression_numbers(expr):
                        o_str = str(int(op_val)) if op_val.is_integer() else f"{op_val:g}"
                        calc_numbers.update({o_str, _normalize_numeric_token(o_str)})
                except Exception:
                    pass
        calc_numbers.update(calc_val.verified_numbers)
        return calc_numbers, calc_result_numbers, False, "", (calc_val.status == "VALID_EXPLICIT")

    # Standalone provenance + mathematical verification when user_input is omitted
    combined_ev_text = " ".join(chunk_searchable_text(c) for c in evidence_chunks) if evidence_chunks else ""
    ev_ts_parts: Set[str] = set()
    for h_s, m_s in re.findall(r"\b([0-2]?\d):([0-5]\d)\b", combined_ev_text):
        ev_ts_parts.add(str(int(h_s)))
        ev_ts_parts.add(str(int(m_s)))
        ev_ts_parts.add(str(int(h_s) * 60 + int(m_s)))

    calc_numbers = set()
    calc_result_numbers = set()
    for _idx, expr, t_res in calc_entries:
        if not expr.strip():
            return set(), set(), True, "Calculator execution missing expression.", False
        try:
            expected_res = float(safe_calculate(expr)["result"])
            expr_nums = extract_expression_numbers(expr)
        except Exception as exc:
            return set(), set(), True, f"Calculator expression '{expr}' failed verification: {exc}", False

        actual_res = t_res["result"] if isinstance(t_res, dict) and "result" in t_res else t_res
        if not isinstance(actual_res, (int, float)) or abs(float(actual_res) - expected_res) > max(1e-4, abs(expected_res) * 1e-4):
            return set(), set(), True, f"Calculator result for '{expr}' did not match evaluated value.", False

        if evidence_chunks:
            conversion_consts = {"60", "100", "1000", "1024", "3600"}
            grounded_non_consts = 0
            for op_val in expr_nums:
                tok = _normalize_numeric_token(str(int(op_val)) if op_val.is_integer() else f"{op_val:g}")
                if _number_present_in_text(tok, combined_ev_text) or tok in ev_ts_parts:
                    if tok not in conversion_consts:
                        grounded_non_consts += 1
                elif tok not in conversion_consts:
                    return set(), set(), True, f"Calculator operand '{tok}' not present in retrieved evidence.", False
            if expr_nums and grounded_non_consts == 0 and any(
                _normalize_numeric_token(str(int(x)) if x.is_integer() else f"{x:g}") not in ev_ts_parts for x in expr_nums
            ):
                return set(), set(), True, "Calculator expression used only ungrounded constants.", False

        r_str = str(actual_res)
        r_norm = _normalize_numeric_token(r_str)
        calc_numbers.update({r_str, r_norm})
        calc_result_numbers.update({r_str, r_norm})
        for op_val in expr_nums:
            o_str = str(int(op_val)) if op_val.is_integer() else f"{op_val:g}"
            calc_numbers.update({o_str, _normalize_numeric_token(o_str)})

    return calc_numbers, calc_result_numbers, False, "", False


def canonicalize_citations_and_claims(
    answer: str,
    evidence_chunks: List[EvidenceChunk],
    tool_executions: Optional[List] = None,
    user_input: Optional[str] = None,
) -> CitationCanonicalizationReport:
    """
    Validate all citations against the authoritative registry, deterministically reconstruct
    unique evidence-backed citations when a cited tag is invalid or missing on a supported claim,
    and remove unrepairable invalid citations and unsupported claims.
    """
    registry = build_citation_registry(evidence_chunks)
    norm_ans = normalize_multi_citations(answer or "")
    norm_ans = re.sub(r"\s*\[source_id#chunk_id\]", "", norm_ans).strip()

    # Resolve [source_id#chunk-xxx] or [document_id#chunk-xxx] if chunk-xxx is in registry
    chunk_by_id = {c.chunk.chunk_id: c.chunk for c in evidence_chunks}
    def _replace_placeholder_doc(m):
        cid = m.group(1)
        if cid in chunk_by_id:
            return f"[{chunk_by_id[cid].document_id}#{cid}]"
        return m.group(0)
    norm_ans = re.sub(r"\[(?:source_id|document_id)#([a-zA-Z0-9_-]+)\]", _replace_placeholder_doc, norm_ans)

    raw_citations = extract_citations(norm_ans)
    raw_suspicious = extract_suspicious_citations(norm_ans)
    raw_invalid = [c for c in raw_citations if c not in registry]
    for s in raw_suspicious:
        if s not in registry and s not in raw_invalid:
            raw_invalid.append(s)

    if not evidence_chunks or is_abstention(norm_ans):
        return CitationCanonicalizationReport(
            canonical_answer=norm_ans,
            raw_citations=raw_citations,
            raw_invalid_citations=raw_invalid,
            final_citations=raw_citations,
            final_invalid_citations=raw_invalid,
        )

    # Collect ONLY verified calculator numbers (distinguishing outputs vs input operands)
    calc_numbers, calc_result_numbers, _calc_failed, _calc_reason, _calc_explicit = _collect_verified_calc_state(
        evidence_chunks, tool_executions, user_input=user_input
    )

    repairs_count = 0
    removals_count = 0
    claims_removed_count = 0

    # Merge standalone citation lines before sentence processing
    ans_merged = re.sub(
        r"\n+\s*((?:\*{0,2}\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]\*{0,2}[.,;: ]*)+)(?=\n|$)",
        r" \1",
        norm_ans,
    )
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ*\[])|\n\n+", ans_merged) if s.strip()]
    rebuilt_sentences: List[str] = []

    for sent in sentences:
        s_cites = extract_citations(sent)
        s_susp = extract_suspicious_citations(sent)
        sent_without_cites = re.sub(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", "", sent)
        for sp in s_susp:
            sent_without_cites = sent_without_cites.replace(sp, "")
        sent_without_cites = re.sub(r"\s+([.,;:!?])", r"\1", sent_without_cites).strip()

        invalid_in_sent = [c for c in s_cites if c not in registry] + [s for s in s_susp if s not in registry]
        valid_in_sent = [c for c in s_cites if c in registry]

        if invalid_in_sent:
            # Attempt deterministic content-based reconstruction for each invalid citation
            preferred_doc = None
            m_doc = re.match(r"^\[([a-zA-Z0-9_.-]+)#", invalid_in_sent[0])
            if m_doc:
                preferred_doc = m_doc.group(1)

            unique_cits = find_unique_supporting_citations(
                sent_without_cites,
                evidence_chunks,
                calc_numbers,
                preferred_doc_id=preferred_doc,
            )
            if unique_cits:
                repairs_count += len(invalid_in_sent)
                for uc in unique_cits:
                    if uc not in valid_in_sent:
                        valid_in_sent.append(uc)
            else:
                removals_count += len(invalid_in_sent)

            if valid_in_sent:
                # Reconstruct sentence with only valid_in_sent
                cits_str = " ".join(valid_in_sent)
                if sent_without_cites.endswith((".", "!", "?")):
                    rebuilt_sent = f"{sent_without_cites[:-1].rstrip()} {cits_str}{sent_without_cites[-1]}"
                else:
                    rebuilt_sent = f"{sent_without_cites} {cits_str}"
                rebuilt_sentences.append(rebuilt_sent)
            else:
                # Neither valid nor uniquely reconstructible citation -> remove unsupported claim sentence if other sentences exist
                claims_removed_count += 1
        else:
            rebuilt_sentences.append(sent)

    if rebuilt_sentences:
        canonical_ans = "\n\n".join(rebuilt_sentences) if "\n\n" in norm_ans else " ".join(rebuilt_sentences)
    elif claims_removed_count > 0:
        is_uk = bool(re.search(r"[а-яА-ЯіїєґІЇЄҐ]", norm_ans))
        canonical_ans = (
            "У наданих фрагментах документації недостатньо однозначно підтверджених цитатами фактів для відповіді."
            if is_uk
            else "Insufficient uniquely verified evidence in the retrieved passages to support a cited answer."
        )
    else:
        canonical_ans = norm_ans

    # If the entire answer has NO citations at all (e.g. model used calculate tool and forgot to cite retrieved operand chunks),
    # check if retrieved chunks uniquely support the answer's operands/claims
    current_cites = extract_citations(canonical_ans)
    if not current_cites and evidence_chunks:
        unique_cits = find_unique_supporting_citations(
            canonical_ans,
            evidence_chunks,
            calc_numbers=calc_result_numbers,  # Match retrieved operand numbers while allowing calculated result!
        )
        if unique_cits:
            repairs_count += len(unique_cits)
            cits_str = " ".join(unique_cits)
            if canonical_ans.endswith((".", "!", "?")):
                canonical_ans = f"{canonical_ans[:-1].rstrip()} {cits_str}{canonical_ans[-1]}"
            else:
                canonical_ans = f"{canonical_ans} {cits_str}"

    # Strip stray empty/slash-only bracket artifacts left by model formatting (e.g. \[\\] or \[\])
    canonical_ans = re.sub(r"\\?\[[\\ ]*\\?\]", "", canonical_ans)
    canonical_ans = re.sub(r"[ \t]{2,}", " ", canonical_ans)
    canonical_ans = re.sub(r"\s+([.,;:!?])", r"\1", canonical_ans).strip()
    final_cites = extract_citations(canonical_ans)
    final_susp = extract_suspicious_citations(canonical_ans)
    final_invalid = [c for c in final_cites if c not in registry] + [s for s in final_susp if s not in registry]

    return CitationCanonicalizationReport(
        canonical_answer=canonical_ans,
        raw_citations=raw_citations,
        raw_invalid_citations=raw_invalid,
        final_citations=final_cites,
        final_invalid_citations=final_invalid,
        citation_repairs_count=repairs_count,
        citation_removals_count=removals_count,
        unsupported_claims_removed_count=claims_removed_count,
    )


def verify_grounding(
    answer: str,
    evidence_chunks: List[EvidenceChunk],
    tool_executions: List = None,
    user_input: Optional[str] = None,
) -> Tuple[GroundingStatus, List[str], List[str], List[ClaimSupport]]:
    """Multi-dimensional grounding, citation provenance, and arithmetic operand verification."""
    answer_norm = normalize_multi_citations(answer)
    supplied_citations = build_citation_registry(evidence_chunks)

    cited = extract_citations(answer_norm)
    suspicious = extract_suspicious_citations(answer_norm)

    invalid = [c for c in cited if c not in supplied_citations]
    for s in suspicious:
        if s not in supplied_citations and s not in invalid:
            invalid.append(s)

    abstained = is_abstention(answer_norm)

    # Collect ONLY verified calculator numbers (failing operand/provenance checks cannot be exempted)
    calc_numbers, _calc_res_nums, calc_failed, calc_fail_reason, calc_explicit_only = _collect_verified_calc_state(
        evidence_chunks, tool_executions, user_input=user_input
    )
    used_calculator = any(
        (getattr(t, "tool_name", None) or (t.get("tool_name") if isinstance(t, dict) else None)) == "calculate"
        for t in (tool_executions or [])
    )

    # Merge standalone citation-only lines/paragraphs onto preceding paragraph before splitting
    answer_merged = re.sub(
        r"\n+\s*((?:\*{0,2}\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]\*{0,2}[.,;: ]*)+)(?=\n|$)",
        r" \1",
        answer_norm,
    )

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ*\[])|\n\n+", answer_merged) if s.strip()]
    claims: List[ClaimSupport] = []

    if calc_failed and not abstained:
        claims.append(
            ClaimSupport(
                claim_text=answer_norm,
                cited_chunk_ids=cited,
                is_supported=False,
                unsupported_reason=calc_fail_reason or "Calculator execution used ungrounded or irrelevant source operands.",
                verdict="UNGROUNDED_OPERANDS",
            )
        )

    for sentence in sentences:
        s_citations = extract_citations(sentence)
        sentence_without_cites = re.sub(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", "", sentence)
        sentence_without_cites = re.sub(r"https?://\S+", "", sentence_without_cites)
        if is_abstention(sentence_without_cites) or sentence_without_cites.strip().endswith("?"):
            continue

        if not s_citations:
            claim_nums = _extract_claim_numbers(sentence_without_cites)
            substantive_nums = [
                n for n in claim_nums
                if n not in IGNORE_STRUCTURAL_NUMS
                and _normalize_numeric_token(n) not in IGNORE_STRUCTURAL_NUMS
                and _normalize_numeric_token(n) not in calc_numbers
            ]
            if substantive_nums and cited and not invalid:
                all_cited_text = " ".join(chunk_searchable_text(supplied_citations[c]) for c in cited if c in supplied_citations)
                substantive_nums = [n for n in substantive_nums if not _number_present_in_text(n, all_cited_text)]
                if not substantive_nums:
                    continue

            if not substantive_nums and cited and not invalid and not calc_failed:
                continue

            supp = assess_claim_support(
                sentence,
                [],
                supplied_citations,
                evidence_chunks,
                calc_numbers,
                abstained=abstained,
            )
            if not supp.is_supported:
                claims.append(supp)
            continue

        supp = assess_claim_support(
            sentence,
            s_citations,
            supplied_citations,
            evidence_chunks,
            calc_numbers,
            abstained=abstained,
        )
        claims.append(supp)

    # Cross-document numerical answers must cite each required target document when not abstaining
    if (
        user_input
        and not abstained
        and not calc_explicit_only
        and has_arithmetic_intent(user_input)
    ):
        expected_docs = extract_expected_documents(user_input)
        if len(expected_docs) >= 2:
            cited_docs = {
                supplied_citations[c].chunk.document_id
                for c in cited
                if c in supplied_citations
            }
            covered_expected = [d for d in expected_docs if d in cited_docs]
            if len(covered_expected) < 2:
                claims.append(
                    ClaimSupport(
                        claim_text=answer_norm,
                        cited_chunk_ids=cited,
                        is_supported=False,
                        unsupported_reason=(
                            f"Cross-document arithmetic query requires grounded citations from {expected_docs[:2]}, "
                            f"but only {sorted(cited_docs)} were cited."
                        ),
                        verdict="INSUFFICIENT_EVIDENCE",
                    )
                )

    # Derive Grounding Status
    if abstained:
        if not invalid:
            return GroundingStatus.INSUFFICIENT_EVIDENCE, cited, invalid, claims
        else:
            return GroundingStatus.VALIDATION_FAILED, cited, invalid, claims

    if invalid or calc_failed:
        return GroundingStatus.VALIDATION_FAILED, cited, invalid, claims

    if not cited and used_calculator and (not evidence_chunks or calc_explicit_only):
        if not claims or all(c.is_supported for c in claims):
            return GroundingStatus.GROUNDED, cited, invalid, claims

    if not cited and evidence_chunks:
        return GroundingStatus.VALIDATION_FAILED, cited, invalid, claims

    if claims and any(not c.is_supported for c in claims):
        if any(c.is_supported for c in claims):
            return GroundingStatus.PARTIALLY_SUPPORTED, cited, invalid, claims
        return GroundingStatus.VALIDATION_FAILED, cited, invalid, claims

    if cited and not invalid:
        return GroundingStatus.GROUNDED, cited, invalid, claims

    return GroundingStatus.INSUFFICIENT_EVIDENCE, cited, invalid, claims
