"""Domain enums for Knowledge Copilot."""
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"


class RetrievalMode(str, Enum):
    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"


class ScoreType(str, Enum):
    COSINE_DISTANCE = "cosine_distance"
    BM25_SCORE = "bm25_score"
    RRF_SCORE = "rrf_score"


class GroundingStatus(str, Enum):
    GROUNDED = "GROUNDED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    EXECUTION_FAILED = "EXECUTION_FAILED"


class ToolCallOrigin(str, Enum):
    MODEL_REQUESTED = "MODEL_REQUESTED"
    POLICY_ENFORCED = "POLICY_ENFORCED"
    RECOVERY_EXECUTED = "RECOVERY_EXECUTED"


class EvidenceQuality(str, Enum):
    CANONICAL_REPORT = "canonical_report"
    QUOTED_MODEL_OUTPUT = "quoted_model_output"
    USER_DOCUMENT = "user_document"
