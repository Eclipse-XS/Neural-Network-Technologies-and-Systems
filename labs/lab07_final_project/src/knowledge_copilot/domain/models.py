"""Domain models and value objects for Knowledge Copilot."""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .enums import DocumentStatus, EvidenceQuality, GroundingStatus, ScoreType, ToolCallOrigin


@dataclass(frozen=True)
class Score:
    retrieval_method: str
    score_type: ScoreType
    score_value: float
    ranking_position: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_method": self.retrieval_method,
            "score_type": self.score_type.value if isinstance(self.score_type, ScoreType) else str(self.score_type),
            "score_value": round(self.score_value, 4),
            "ranking_position": self.ranking_position,
        }


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    ordinal: int
    text: str
    section: str
    token_count: int
    sha256: str
    evidence_quality: EvidenceQuality = EvidenceQuality.USER_DOCUMENT
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def citation_id(self) -> str:
        return f"[{self.document_id}#{self.chunk_id}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "ordinal": self.ordinal,
            "text": self.text,
            "section": self.section,
            "token_count": self.token_count,
            "sha256": self.sha256,
            "evidence_quality": self.evidence_quality.value if isinstance(self.evidence_quality, EvidenceQuality) else str(self.evidence_quality),
            "citation": self.citation_id,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class EvidenceChunk:
    chunk: Chunk
    score: Score

    @property
    def source_id(self) -> str:
        return self.chunk.document_id

    @property
    def document_id(self) -> str:
        return self.chunk.document_id

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def section(self) -> str:
        return self.chunk.section

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def citation(self) -> str:
        return self.chunk.citation_id

    @property
    def evidence_quality(self) -> str:
        return self.chunk.evidence_quality.value if isinstance(self.chunk.evidence_quality, EvidenceQuality) else str(self.chunk.evidence_quality)

    @property
    def filename(self) -> str:
        return self.chunk.metadata.get("filename", f"{self.document_id}.md")

    @property
    def lab_number(self) -> str:
        return str(self.chunk.metadata.get("lab_number", ""))

    @property
    def cosine_distance(self) -> float:
        # Backward-compatibility alias for legacy evaluators
        return self.score.score_value

    def to_dict(self) -> Dict[str, Any]:
        res = self.chunk.to_dict()
        res.update(self.score.to_dict())
        res["source_id"] = self.document_id
        res["filename"] = self.filename
        res["lab_number"] = self.lab_number
        res["cosine_distance"] = self.score.score_value
        return res


@dataclass
class Document:
    document_id: str
    filename: str
    content_type: str
    sha256: str
    size_bytes: int
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    collection: str = "default"
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "content_type": self.content_type,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "status": self.status.value if isinstance(self.status, DocumentStatus) else str(self.status),
            "chunk_count": self.chunk_count,
            "collection": self.collection,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


@dataclass
class ToolExecutionRecord:
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    origin: ToolCallOrigin = ToolCallOrigin.MODEL_REQUESTED
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "origin": self.origin.value if isinstance(self.origin, ToolCallOrigin) else str(self.origin),
            "execution_time_ms": round(self.execution_time_ms, 2),
        }


@dataclass
class ClaimSupport:
    claim_text: str
    cited_chunk_ids: List[str]
    is_supported: bool
    unsupported_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    turn_id: str
    session_id: str
    user_input: str
    raw_answer: str
    canonical_answer: str
    grounding_status: GroundingStatus
    tool_executions: List[ToolExecutionRecord] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    claims: List[ClaimSupport] = field(default_factory=list)
    evidence: List[EvidenceChunk] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    runtime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "user_input": self.user_input,
            "raw_answer": self.raw_answer,
            "canonical_answer": self.canonical_answer,
            "grounding_status": self.grounding_status.value if isinstance(self.grounding_status, GroundingStatus) else str(self.grounding_status),
            "tools_used": [t.to_dict() for t in self.tool_executions],
            "citations": self.citations,
            "invalid_citations": self.invalid_citations,
            "claims": [c.to_dict() for c in self.claims],
            "evidence": [e.to_dict() for e in self.evidence],
            "created_at": self.created_at,
            "runtime_seconds": round(self.runtime_seconds, 3),
        }


@dataclass
class Session:
    session_id: str
    title: str = "New Conversation"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turns: List[Turn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": len(self.turns),
            "turns": [t.to_dict() for t in self.turns],
        }
