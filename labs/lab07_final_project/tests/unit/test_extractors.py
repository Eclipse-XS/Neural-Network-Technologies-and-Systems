"""Unit tests for document file extractors and validation (UC-01)."""
import pytest
import pymupdf

from labs.lab07_final_project.src.knowledge_copilot.domain.exceptions import (
    DocumentValidationError,
    ExtractionError,
    OversizedDocumentError,
    UnsupportedFormatError,
)
from labs.lab07_final_project.src.knowledge_copilot.rag.extractors import (
    extract_text_from_bytes,
    sanitize_filename,
    validate_file_content,
)


def test_sanitize_filename_rejects_path_traversal():
    assert sanitize_filename("doc.md") == "doc.md"
    assert sanitize_filename("subfolder/doc.md") == "doc.md"
    assert sanitize_filename("subfolder\\doc.md") == "doc.md"
    assert sanitize_filename("../../etc/passwd") == "passwd"
    
    with pytest.raises(DocumentValidationError):
        sanitize_filename("")
        
    with pytest.raises(DocumentValidationError):
        sanitize_filename("   ")

    with pytest.raises(DocumentValidationError):
        sanitize_filename("evil\0file.md")


def test_validate_file_content_enforces_rules():
    content = b"# Architecture Overview\nThis is a sample technical document."
    name, sha, size = validate_file_content(content, "architecture.md")
    assert name == "architecture.md"
    assert size == len(content)
    assert len(sha) == 64

    # Empty content
    with pytest.raises(DocumentValidationError, match="is empty"):
        validate_file_content(b"", "empty.txt")

    # Unsupported format
    with pytest.raises(UnsupportedFormatError, match="Unsupported file format"):
        validate_file_content(b"ELF...", "binary.exe")

    # Oversized document
    with pytest.raises(OversizedDocumentError, match="exceeds the maximum allowed limit"):
        validate_file_content(b"A" * 100, "large.txt", max_size_bytes=50)


def test_extract_markdown_and_text():
    md_content = b"# Specification\n## Section 1\nVerified text."
    txt = extract_text_from_bytes(md_content, "spec.md")
    assert "Specification" in txt
    assert "Verified text." in txt

    txt_content = "Текстовий документ з кодуванням UTF-8.".encode("utf-8")
    txt_extracted = extract_text_from_bytes(txt_content, "notes.txt")
    assert "Текстовий документ" in txt_extracted


def test_extract_text_from_valid_pdf():
    # Create a minimal in-memory text PDF using PyMuPDF
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Neural Network Architecture Specification")
    pdf_bytes = doc.tobytes()
    doc.close()

    extracted = extract_text_from_bytes(pdf_bytes, "spec.pdf")
    assert "Neural Network Architecture Specification" in extracted
    assert "## Page 1" in extracted


def test_scanned_pdf_without_text_layer_raises_explicit_error():
    # Create an empty-page PDF (simulating scanned image without OCR)
    doc = pymupdf.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()

    with pytest.raises(ExtractionError, match="Scanned or image-only PDF: 'empty.pdf' contains no extractable text layer"):
        extract_text_from_bytes(pdf_bytes, "empty.pdf")
