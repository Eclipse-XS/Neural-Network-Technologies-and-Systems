"""Knowledge Copilot RAG layer."""
from .extractors import extract_text_from_bytes, sanitize_filename, validate_file_content
from .splitters import build_chunks_from_document
from .lexical import BM25Retriever
from .hybrid import reciprocal_rank_fusion

def __getattr__(name: str):
    if name == "CopilotEmbeddings":
        from .embeddings import CopilotEmbeddings
        return CopilotEmbeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "extract_text_from_bytes",
    "sanitize_filename",
    "validate_file_content",
    "build_chunks_from_document",
    "CopilotEmbeddings",
    "BM25Retriever",
    "reciprocal_rank_fusion",
]
