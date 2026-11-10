"""Pydantic schemas for Knowledge Copilot API contracts."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    content_type: str
    sha256: str
    size_bytes: int
    status: str
    chunk_count: int
    collection: str
    error_message: Optional[str] = None
    created_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="User question or query.")
    session_id: Optional[str] = Field(None, description="Optional existing session ID.")
    retrieval_mode: Optional[str] = Field("hybrid", description="dense, lexical, or hybrid")


class EvidenceChunkSchema(BaseModel):
    source_id: str
    document_id: str
    chunk_id: str
    section: str
    text: str
    citation: str
    retrieval_method: str
    score_type: str
    score_value: float
    ranking_position: int


class ToolRecordSchema(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    origin: str
    execution_time_ms: float


class ChatResponse(BaseModel):
    turn_id: str
    session_id: str
    query: str
    answer: str
    grounding_status: str
    citations: List[str]
    invalid_citations: List[str]
    evidence: List[EvidenceChunkSchema]
    tools_used: List[ToolRecordSchema]
    runtime_seconds: float


class SessionTurnSchema(BaseModel):
    turn_id: str
    user_input: str
    canonical_answer: str
    grounding_status: str
    citations: List[str]
    created_at: str
    runtime_seconds: float


class SessionResponse(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    turn_count: int
    turns: List[SessionTurnSchema]


class HealthResponse(BaseModel):
    status: str
    version: str
    redis_connected: bool
    sqlite_connected: bool
    llm_available: bool
