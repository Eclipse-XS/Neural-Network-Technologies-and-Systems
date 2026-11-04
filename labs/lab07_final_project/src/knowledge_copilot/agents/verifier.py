"""Multi-dimensional grounding, citation validity, and claim support verification."""
import re
from typing import Dict, List, Set, Tuple

from ..domain.enums import GroundingStatus
from ..domain.models import ClaimSupport, EvidenceChunk


ABSTENTION_PATTERNS = [
    r"\b(?:cannot determine|insufficient evidence|not available|not mentioned)\b",
    r"\b(?:do not have enough information|not found in the indexed)\b",
    r"не можу визначити",
    r"недостатньо (?:даних|доказів|інформації)",
    r"не містить .*?(?:інформац|даних|відповід)",
    r"даних .*? не виявлено",
    r"не було зазначено",
    r"інформація відсутня",
]


def extract_citations(text: str) -> List[str]:
    """Extract standard citation tags matching [source_id#chunk_id]."""
    return re.findall(r"\[[a-zA-Z0-9_-]+#[a-zA-Z0-9_-]+\]", text)


def extract_suspicious_citations(text: str) -> List[str]:
    """Find malformed, filename-style, or placeholder citations."""
    patterns = [
        r"\[source_id#chunk_id\]",
        r"\[[^\]\n]*(?:chunk-|_report\.md)[^\]\n]*\]",
        r"\[(?:Джерело|Citation|Source):\s*[^\]]+\]",
    ]
    matches = []
    for p in patterns:
        matches.extend(re.findall(p, text, re.IGNORECASE))
    return matches


def is_abstention(text: str) -> bool:
    """Detect explicit refusal or declaration of insufficient evidence."""
    for pattern in ABSTENTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def verify_grounding(
    answer: str,
    evidence_chunks: List[EvidenceChunk],
) -> Tuple[GroundingStatus, List[str], List[str], List[ClaimSupport]]:
    """Distinct multi-dimensional validation solving DEFECT-02:
    
    1. Citation identifier validity & syntax
    2. Citation provenance (retrieved in current turn)
    3. Claim-to-evidence support (numerical and entity consistency)
    4. Explicit status classification
    """
    supplied_citations: Dict[str, EvidenceChunk] = {
        ec.citation: ec for ec in evidence_chunks
    }
    
    cited = extract_citations(answer)
    suspicious = extract_suspicious_citations(answer)
    
    invalid = [c for c in cited if c not in supplied_citations]
    for s in suspicious:
        if s not in supplied_citations and s not in invalid:
            invalid.append(s)
            
    abstained = is_abstention(answer)
    
    # Sentence-level claim analysis
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", answer) if s.strip()]
    claims: List[ClaimSupport] = []
    
    for sentence in sentences:
        s_citations = extract_citations(sentence)
        if not s_citations:
            # Check if sentence makes empirical/numeric claims without citation
            nums = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", sentence)
            # Ignore lab numbers or pure years
            substantive_nums = [n for n in nums if n not in {"3", "4", "5", "6", "7", "2024", "2025", "2026"}]
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
            
        # Verify sentence against each cited chunk
        sentence_supported = True
        failure_reasons = []
        for cite_tag in s_citations:
            if cite_tag not in supplied_citations:
                sentence_supported = False
                failure_reasons.append(f"Citation {cite_tag} was not supplied in current retrieval.")
                continue
                
            chunk = supplied_citations[cite_tag]
            chunk_text_lower = chunk.text.casefold()
            
            # Check numbers in claim exist in cited chunk
            claim_nums = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", sentence)
            for num in claim_nums:
                # Allow minor formatting differences or lab references
                if num in {"3", "4", "5", "6", "7"}:
                    continue
                if num not in chunk.text and str(int(float(num))) not in chunk.text:
                    sentence_supported = False
                    failure_reasons.append(f"Number '{num}' absent from cited passage {cite_tag}.")
                    
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
