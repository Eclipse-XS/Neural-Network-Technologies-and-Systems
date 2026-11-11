"""Multi-dimensional grounding, citation validity, and claim support verification."""
import re
from typing import Dict, List, Set, Tuple

from ..domain.enums import GroundingStatus
from ..domain.models import ClaimSupport, EvidenceChunk


ABSTENTION_PATTERNS = [
    r"\b(?:cannot determine|insufficient evidence|not available|not mentioned|unable to determine|cannot answer|do not cover|not covered)\b",
    r"\b(?:do not have enough information|not found in the indexed|do(?:es)? not contain|no (?:specific |relevant |direct )?(?:benchmark |information|evidence|data|mention|details|instructions|guidance))\b",
    r"\b(?:cannot retrieve|unable to find|not explicitly (?:list|mention|state|report|describe|specify|detail|provide|configure)|please clarify which)\b",
    r"\b(?:is not explicitly configured|is not configured|are not configured)\b",
    r"\b(?:outside (?:the )?(?:scope|indexed|documentation))\b",
    r"не можу визначити",
    r"недостатньо (?:даних|доказів|інформації)",
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


def strip_markdown(text: str) -> str:
    """Strip markdown formatting characters for robust text inspection."""
    return re.sub(r"[*_`#]", "", text or "")


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
    # Also flag any bracketed chunk reference that is NOT a valid [doc#chunk-...] tag
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
    cleaned = num_str.replace(",", "").strip()
    try:
        val = float(cleaned)
        if val.is_integer():
            return str(int(val))
        return f"{val:g}"
    except ValueError:
        return cleaned


def _number_present_in_text(num_str: str, target_text: str) -> bool:
    """Check whether a numeric token appears in target_text in raw, comma-stripped, or integer form."""
    if num_str in target_text:
        return True
    cleaned = num_str.replace(",", "")
    if cleaned in target_text:
        return True
    norm = _normalize_numeric_token(cleaned)
    if norm in target_text:
        return True
    target_no_commas = re.sub(r"(?<=\d),(?=\d{3}\b)", "", target_text)
    return cleaned in target_no_commas or norm in target_no_commas


def verify_grounding(
    answer: str,
    evidence_chunks: List[EvidenceChunk],
    tool_executions: List = None,
) -> Tuple[GroundingStatus, List[str], List[str], List[ClaimSupport]]:
    """Distinct multi-dimensional validation solving DEFECT-02:
    
    1. Citation identifier validity & syntax (with bracket grouping normalization)
    2. Citation provenance (retrieved in current turn)
    3. Claim-to-evidence support (numerical and technology premise consistency)
    4. Explicit status classification (including calculator-backed and refusal turns)
    """
    answer_norm = normalize_multi_citations(answer)
    supplied_citations: Dict[str, EvidenceChunk] = {}
    for ec in evidence_chunks:
        supplied_citations[ec.citation] = ec
        # Also allow matching by filename stem or source_id if identical chunk_id
        c_id = ec.chunk.chunk_id
        d_id = ec.chunk.document_id
        supplied_citations[f"[{d_id}#{c_id}]"] = ec
        if ec.chunk.metadata and "source_id" in ec.chunk.metadata:
            supplied_citations[f"[{ec.chunk.metadata['source_id']}#{c_id}]"] = ec

    cited = extract_citations(answer_norm)
    suspicious = extract_suspicious_citations(answer_norm)

    invalid = [c for c in cited if c not in supplied_citations]
    for s in suspicious:
        if s not in supplied_citations and s not in invalid:
            invalid.append(s)

    abstained = is_abstention(answer_norm)

    # Collect calculator tool results if present
    calc_numbers: Set[str] = set()
    used_calculator = False
    if tool_executions:
        for t in tool_executions:
            t_name = getattr(t, "tool_name", None) or (t.get("tool_name") if isinstance(t, dict) else None)
            if t_name == "calculate":
                used_calculator = True
                t_res = getattr(t, "result", None) or (t.get("result") if isinstance(t, dict) else None)
                if isinstance(t_res, dict) and "result" in t_res:
                    res_val = str(t_res["result"])
                    calc_numbers.add(res_val)
                    calc_numbers.add(_normalize_numeric_token(res_val))
                t_args = getattr(t, "arguments", None) or (t.get("arguments") if isinstance(t, dict) else None)
                if isinstance(t_args, dict) and "expression" in t_args:
                    for n in re.findall(r"\d+(?:\.\d+)?", str(t_args["expression"])):
                        calc_numbers.add(n)
                        calc_numbers.add(_normalize_numeric_token(n))

    # Merge standalone citation-only lines/paragraphs onto preceding paragraph before splitting
    answer_merged = re.sub(
        r"\n+\s*((?:\*{0,2}\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]\*{0,2}[.,;: ]*)+)(?=\n|$)",
        r" \1",
        answer_norm,
    )

    # Split into paragraphs / logical blocks so bulleted lists ending with a citation share context
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯІЇЄҐ*\[])|\n\n+", answer_merged) if s.strip()]
    claims: List[ClaimSupport] = []

    # Ignore structural identifiers, ports, HTTP codes, dimensions, and model sizes
    ignore_nums = {
        "001", "1", "2", "3", "4", "5", "6", "7", "2024", "2025", "2026",
        "1234", "6379", "6380", "8000", "8080", "200", "201", "400", "404", "422", "500", "503",
        "768", "384", "1536", "0.5", "8", "70",
    }

    all_evidence_lower = " ".join(c.text for c in evidence_chunks).lower() if evidence_chunks else ""

    for sentence in sentences:
        s_citations = extract_citations(sentence)
        # Strip citations, URLs, and markdown formatting before extracting numbers
        sentence_without_cites = re.sub(r"\[[a-zA-Z0-9_.-]+#[a-zA-Z0-9_.-]+\]", "", sentence)
        sentence_without_cites = re.sub(r"https?://\S+", "", sentence_without_cites)
        # Ignore Uncertainty / follow-up question sentences that make no factual assertions
        if is_abstention(sentence_without_cites) or sentence_without_cites.strip().endswith("?"):
            continue

        # Enforce answer-wide technology premise validation when evidence was supplied
        if evidence_chunks and not abstained:
            s_lower_global = strip_markdown(sentence_without_cites).lower()
            absent_techs = [
                tech for tech in KNOWN_TECHNOLOGIES
                if re.search(r"\b" + re.escape(tech) + r"\b", s_lower_global)
                and not re.search(r"\b" + re.escape(tech) + r"\b", all_evidence_lower)
            ]
            if absent_techs and not s_citations:
                claims.append(
                    ClaimSupport(
                        claim_text=sentence,
                        cited_chunk_ids=[],
                        is_supported=False,
                        unsupported_reason=f"Subject '{absent_techs[0]}' absent from retrieved evidence.",
                    )
                )
                continue

        if not s_citations:
            # Check if sentence makes empirical/numeric claims without citation
            nums = re.findall(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?", sentence_without_cites)
            substantive_nums = [
                n for n in nums
                if n not in ignore_nums
                and _normalize_numeric_token(n) not in ignore_nums
                and _normalize_numeric_token(n) not in calc_numbers
            ]
            # If the whole answer has valid citations and the number exists in any cited chunk, don't fail a header/summary line
            if substantive_nums and cited and not invalid:
                all_cited_text = " ".join(supplied_citations[c].text for c in cited if c in supplied_citations)
                substantive_nums = [n for n in substantive_nums if not _number_present_in_text(n, all_cited_text)]

            if substantive_nums and not abstained:
                claims.append(
                    ClaimSupport(
                        claim_text=sentence,
                        cited_chunk_ids=[],
                        is_supported=False,
                        unsupported_reason="Factual claim with numbers lacks supporting citation.",
                    )
                )
            continue

        # Verify sentence against combined text of all cited chunks in this sentence/block
        sentence_supported = True
        failure_reasons = []
        valid_sentence_chunks = []
        for cite_tag in s_citations:
            if cite_tag not in supplied_citations:
                sentence_supported = False
                failure_reasons.append(f"Citation {cite_tag} was not supplied in current retrieval.")
            else:
                valid_sentence_chunks.append(supplied_citations[cite_tag])

        if valid_sentence_chunks:
            combined_cited_text = " ".join(c.text for c in valid_sentence_chunks)
            combined_cited_lower = combined_cited_text.lower()
            s_lower = sentence_without_cites.lower()

            # 1. Technology premise validation: if claim sentence mentions a technology, it must appear in cited evidence
            for tech in KNOWN_TECHNOLOGIES:
                if re.search(r"\b" + re.escape(tech) + r"\b", s_lower):
                    if not re.search(r"\b" + re.escape(tech) + r"\b", combined_cited_lower):
                        sentence_supported = False
                        failure_reasons.append(f"Subject '{tech}' absent from cited passage(s) {s_citations}.")

            # 2. Numeric validation
            claim_nums = re.findall(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?", sentence_without_cites)
            for num in claim_nums:
                if (
                    num in ignore_nums
                    or _normalize_numeric_token(num) in ignore_nums
                    or _normalize_numeric_token(num) in calc_numbers
                ):
                    continue
                if not _number_present_in_text(num, combined_cited_text):
                    sentence_supported = False
                    failure_reasons.append(f"Number '{num}' absent from cited passage(s) {s_citations}.")

        claims.append(
            ClaimSupport(
                claim_text=sentence,
                cited_chunk_ids=s_citations,
                is_supported=sentence_supported,
                unsupported_reason="; ".join(failure_reasons) if failure_reasons else None,
            )
        )

    # Derive Grounding Status
    if abstained:
        if not invalid:
            return GroundingStatus.INSUFFICIENT_EVIDENCE, cited, invalid, claims
        else:
            return GroundingStatus.VALIDATION_FAILED, cited, invalid, claims

    if invalid:
        return GroundingStatus.VALIDATION_FAILED, cited, invalid, claims

    # Pure calculator turn with no document retrieval required
    if not cited and not evidence_chunks and used_calculator:
        if not claims or all(c.is_supported for c in claims):
            return GroundingStatus.GROUNDED, cited, invalid, claims

    if not cited and evidence_chunks:
        # Expected citations for retrieved evidence
        return GroundingStatus.VALIDATION_FAILED, cited, invalid, claims

    if claims and any(not c.is_supported for c in claims):
        if any(c.is_supported for c in claims):
            return GroundingStatus.PARTIALLY_SUPPORTED, cited, invalid, claims
        return GroundingStatus.VALIDATION_FAILED, cited, invalid, claims

    if cited and not invalid:
        return GroundingStatus.GROUNDED, cited, invalid, claims

    return GroundingStatus.INSUFFICIENT_EVIDENCE, cited, invalid, claims
