"""File extraction and validation for Knowledge Copilot."""
import hashlib
import os
import re
from pathlib import Path
from typing import Tuple

from ..domain.exceptions import (
    ExtractionError,
    OversizedDocumentError,
    UnsupportedFormatError,
    DocumentValidationError,
)

MAX_DEFAULT_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MiB
ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf"}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename against path traversal and forbidden characters."""
    if not filename or not isinstance(filename, str):
        raise DocumentValidationError("Filename must be a non-empty string.")
    
    # Strip directory components (prevents path traversal)
    clean_name = os.path.basename(filename.replace("\\", "/"))
    # Reject null bytes or control characters
    if "\0" in clean_name or re.search(r"[\x00-\x1f\x7f]", clean_name):
        raise DocumentValidationError(f"Invalid characters in filename: {filename}")
    
    # Check for empty or reserved names
    clean_name = clean_name.strip()
    if not clean_name or clean_name in {".", ".."}:
        raise DocumentValidationError(f"Invalid filename: {filename}")
    
    return clean_name


def validate_file_content(data: bytes, filename: str, max_size_bytes: int = MAX_DEFAULT_FILE_SIZE_BYTES) -> Tuple[str, str, int]:
    """Validate raw file bytes and extension, return (sanitized_filename, sha256, size_bytes)."""
    clean_name = sanitize_filename(filename)
    suffix = Path(clean_name).suffix.lower()
    
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file format '{suffix}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    
    size_bytes = len(data)
    if size_bytes == 0:
        raise DocumentValidationError(f"File '{clean_name}' is empty (0 bytes).")
    
    if size_bytes > max_size_bytes:
        raise OversizedDocumentError(
            f"File '{clean_name}' size {size_bytes} bytes exceeds the maximum allowed limit of {max_size_bytes} bytes."
        )
    
    sha256_hash = hashlib.sha256(data).hexdigest()
    return clean_name, sha256_hash, size_bytes


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Extract clean text content from supported file bytes."""
    clean_name, _, _ = validate_file_content(data, filename)
    suffix = Path(clean_name).suffix.lower()
    
    if suffix in {".md", ".txt"}:
        for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1251"):
            try:
                text = data.decode(encoding)
                if text.strip():
                    return text
            except UnicodeDecodeError:
                continue
        raise ExtractionError(f"Failed to decode text from '{clean_name}' using supported encodings.")
    
    if suffix == ".pdf":
        try:
            import pymupdf  # PyMuPDF
            doc = pymupdf.open(stream=data, filetype="pdf")
            extracted_pages = []
            
            for page_idx, page in enumerate(doc):
                page_text = page.get_text("text").strip()
                if page_text:
                    extracted_pages.append(f"## Page {page_idx + 1}\n\n{page_text}")
            
            doc.close()
            
            full_text = "\n\n".join(extracted_pages).strip()
            if not full_text:
                raise ExtractionError(
                    f"Scanned or image-only PDF: '{clean_name}' contains no extractable text layer. OCR required."
                )
            return full_text
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"Error parsing PDF '{clean_name}': {type(exc).__name__}: {exc}") from exc
    
    raise UnsupportedFormatError(f"Extractor not implemented for suffix '{suffix}'")
