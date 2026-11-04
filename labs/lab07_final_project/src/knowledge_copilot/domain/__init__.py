"""Knowledge Copilot Domain layer."""
from .enums import DocumentStatus, EvidenceQuality, GroundingStatus, RetrievalMode, ScoreType, ToolCallOrigin
from .models import ClaimSupport, Chunk, Document, EvidenceChunk, Score, Session, ToolExecutionRecord, Turn
from .exceptions import (
    CopilotDomainError,
    DocumentNotFoundError,
    DocumentValidationError,
    ExtractionError,
    IndexingError,
    OversizedDocumentError,
    SessionConcurrencyError,
    SessionNotFoundError,
    ToolExecutionError,
    UnsupportedFormatError,
)

__all__ = [
    "DocumentStatus",
    "EvidenceQuality",
    "GroundingStatus",
    "RetrievalMode",
    "ScoreType",
    "ToolCallOrigin",
    "ClaimSupport",
    "Chunk",
    "Document",
    "EvidenceChunk",
    "Score",
    "Session",
    "ToolExecutionRecord",
    "Turn",
    "CopilotDomainError",
    "DocumentNotFoundError",
    "DocumentValidationError",
    "ExtractionError",
    "IndexingError",
    "OversizedDocumentError",
    "SessionConcurrencyError",
    "SessionNotFoundError",
    "ToolExecutionError",
    "UnsupportedFormatError",
]
