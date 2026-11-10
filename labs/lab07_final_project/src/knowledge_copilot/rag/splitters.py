"""Structure-aware and token-bounded chunking."""
import hashlib
import json
import re
from typing import Callable, List, Optional

from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from ..domain.enums import EvidenceQuality
from ..domain.models import Chunk, Document


def build_chunks_from_document(
    doc: Document,
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
    max_tokens: int = 240,
    token_counter: Optional[Callable[[str], int]] = None,
) -> List[Chunk]:
    """Chunk document text with heading awareness and strict token limits.
    
    Zero silently truncated embedding inputs: if a token_counter is supplied,
    any chunk exceeding max_tokens is deterministically subdivided.
    """
    if max_tokens < 32:
        raise ValueError("Effective embedding token budget must be at least 32 tokens.")
    
    headers = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    
    # Check if text contains markdown headings
    has_headers = bool(re.search(r"^#{1,3}\s+", text, re.MULTILINE))
    if has_headers:
        sections = headers.split_text(text)
    else:
        sections = [LCDocument(page_content=text, metadata={"h1": doc.filename})]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n- ", "\n* ", "\n1. ", "\n| ", "\n", ". ", " "],
    )
    
    candidates = splitter.split_documents(sections)
    bounded_docs: List[LCDocument] = []
    
    for c_doc in candidates:
        heading = " / ".join(c_doc.metadata[k] for k in ("h1", "h2", "h3") if k in c_doc.metadata)
        if heading and heading.casefold() not in c_doc.page_content[:200].casefold():
            c_doc = LCDocument(
                page_content=f"## {heading}\n{c_doc.page_content}",
                metadata=c_doc.metadata,
            )
        
        pending = [c_doc]
        while pending:
            curr = pending.pop(0)
            if token_counter is None or token_counter(curr.page_content) <= max_tokens:
                bounded_docs.append(curr)
                continue
            
            # Semantic cut avoiding mid-word or mid-sentence slicing
            txt = curr.page_content
            # Candidate semantic separators in order of preference
            split_candidates = []
            for sep in ["\n\n", "\n- ", "\n* ", "\n1. ", "\n| ", "\n", ". "]:
                for m in re.finditer(re.escape(sep), txt):
                    pos = m.end() if sep in [". ", "\n\n"] else m.start()
                    if 0 < pos < len(txt):
                        split_candidates.append(pos)
            
            split_candidates = sorted(set(split_candidates))
            best_cut = None
            if token_counter is not None:
                for pos in reversed(split_candidates):
                    cand_left = txt[:pos].strip()
                    if cand_left and token_counter(cand_left) <= max_tokens and len(cand_left) >= 40:
                        best_cut = pos
                        break
            
            if best_cut is None:
                midpoint = len(txt) // 2
                best_cut = max(
                    txt.rfind("\n", 0, midpoint),
                    txt.rfind(". ", 0, midpoint),
                    txt.rfind(" ", 0, midpoint),
                )
                if best_cut < 1:
                    best_cut = midpoint
            
            left = txt[:best_cut].strip()
            
            # Snap overlap backward to whitespace to guarantee clean word boundaries
            raw_start = max(0, best_cut - chunk_overlap)
            while raw_start > 0 and not txt[raw_start - 1].isspace():
                raw_start -= 1
            right = txt[raw_start:].strip()
            
            if not left or not right or left == txt or right == txt:
                # Force clean cut at midpoint on whitespace if identical
                ws = txt.rfind(" ", 0, len(txt) // 2)
                mid = ws if ws > 0 else len(txt) // 2
                left = txt[:mid].strip()
                right = txt[mid:].strip()
            
            heading_prefix = f"## {heading}\n" if heading else ""
            if heading_prefix and not right.startswith(heading_prefix):
                right = f"{heading_prefix}{right}"

            pending[:0] = [
                LCDocument(page_content=left, metadata=curr.metadata),
                LCDocument(page_content=right, metadata=curr.metadata),
            ]
    
    chunks: List[Chunk] = []
    for ordinal, b_doc in enumerate(bounded_docs):
        section = (
            " / ".join(b_doc.metadata[k] for k in ("h1", "h2", "h3") if k in b_doc.metadata)
            or b_doc.metadata.get("h1", "General")
        )
        normalized_content = " ".join(b_doc.page_content.split())
        digest = hashlib.sha256(
            json.dumps(
                [doc.sha256, section, normalized_content, ordinal],
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()[:16]
        
        chunk_id = f"chunk-{digest}"
        tok_count = token_counter(b_doc.page_content) if token_counter else len(b_doc.page_content.split())
        
        # Determine evidence quality
        quality = EvidenceQuality.USER_DOCUMENT
        if doc.collection == "coursework" or doc.document_id.startswith("lab"):
            transcript_sections = (
                "system instructions", "дизайн експерименту", "підсумкова таблиця",
                "багатокрокове виконання", "додаткова арифметична", "керована діагностика",
                "без інструментів", "невідома інформація",
                "ефективність рішень", "фактичні помилки", "зауваження до кожної",
                "результати", "comparison",
            )
            raw_output_markers = (
                "session a,", "session b,", "to confirm, i will check", "otherwise, based on",
                "final answer:", "the sum of the inferred", "lab_4_results.csv",
            )
            is_quoted = doc.document_id.startswith("lab06") and (
                any(marker in section.casefold() for marker in transcript_sections)
                or any(marker in b_doc.page_content.casefold() for marker in raw_output_markers)
            )
            quality = EvidenceQuality.QUOTED_MODEL_OUTPUT if is_quoted else EvidenceQuality.CANONICAL_REPORT

        metadata = {
            "source_id": doc.document_id,
            "filename": doc.filename,
            "section": section,
            "ordinal": ordinal,
            "collection": doc.collection,
            **b_doc.metadata,
        }
        if "lab_number" in doc.metadata:
            metadata["lab_number"] = doc.metadata["lab_number"]
            
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=doc.document_id,
                ordinal=ordinal,
                text=b_doc.page_content,
                section=section,
                token_count=tok_count,
                sha256=hashlib.sha256(b_doc.page_content.encode("utf-8")).hexdigest(),
                evidence_quality=quality,
                metadata=metadata,
            )
        )
    
    return chunks
