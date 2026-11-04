"""Unit tests for structure-aware, zero-truncation chunking."""
import pytest
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import DocumentStatus, EvidenceQuality
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Document
from labs.lab07_final_project.src.knowledge_copilot.rag.splitters import build_chunks_from_document


def test_structure_aware_chunking_preserves_headings():
    doc = Document(
        document_id="doc_test1",
        filename="system.md",
        content_type="md",
        sha256="abc12345",
        size_bytes=500,
        status=DocumentStatus.INDEXED,
    )
    markdown_text = """# System Overview
Introductory text.

## Architecture
Detailed description of modular monolith.

### Ingestion Service
Handles parsing and chunking.
"""
    chunks = build_chunks_from_document(
        doc=doc,
        text=markdown_text,
        chunk_size=200,
        chunk_overlap=20,
        max_tokens=100,
    )
    assert len(chunks) >= 2
    sections = [c.section for c in chunks]
    assert any("Architecture" in s for s in sections)
    assert any("Ingestion Service" in s for s in sections)
    for c in chunks:
        assert c.document_id == "doc_test1"
        assert c.chunk_id.startswith("chunk-")
        assert c.citation_id == f"[doc_test1#{c.chunk_id}]"


def test_zero_silently_truncated_inputs():
    """Ensure oversized sections are deterministically subdivided so none exceed token budget."""
    doc = Document(
        document_id="doc_large",
        filename="long.md",
        content_type="md",
        sha256="deadbeef",
        size_bytes=4000,
        status=DocumentStatus.INDEXED,
    )
    long_section = "Word " * 600  # 600 words
    markdown_text = f"# Long Document\n\n## Big Section\n{long_section}"
    
    # Token counter counts words
    token_counter = lambda s: len(s.split())
    max_tokens = 50

    chunks = build_chunks_from_document(
        doc=doc,
        text=markdown_text,
        chunk_size=300,
        chunk_overlap=30,
        max_tokens=max_tokens,
        token_counter=token_counter,
    )

    # Acceptance requirement: ZERO silently truncated embedding inputs
    for c in chunks:
        assert token_counter(c.text) <= max_tokens, f"Chunk {c.chunk_id} exceeded token limit {max_tokens}"
