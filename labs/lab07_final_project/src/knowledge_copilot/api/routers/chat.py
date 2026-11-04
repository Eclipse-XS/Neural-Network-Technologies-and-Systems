"""Conversational chat and session retrieval routes."""
import uuid
from fastapi import APIRouter, Depends, HTTPException, status

from ...domain.enums import RetrievalMode
from ..dependencies import get_query_service, get_session_service
from ..schemas import (
    ChatRequest,
    ChatResponse,
    EvidenceChunkSchema,
    SessionResponse,
    SessionTurnSchema,
    ToolRecordSchema,
)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    query_service=Depends(get_query_service),
    session_service=Depends(get_session_service),
):
    """Execute grounded RAG or conversational agent turn."""
    session_id = request.session_id or f"sess_{uuid.uuid4().hex[:12]}"
    
    # Ensure session exists
    session = session_service.get_session(session_id)
    if not session:
        session = session_service.create_session(session_id)
        
    mode = None
    if request.retrieval_mode:
        try:
            mode = RetrievalMode(request.retrieval_mode.lower())
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid retrieval mode '{request.retrieval_mode}'. Expected 'dense', 'lexical', or 'hybrid'.",
            )
            
    turn = await query_service.execute_query(
        query=request.query,
        session_id=session_id,
        retrieval_mode=mode,
    )
    
    evidence_schemas = [
        EvidenceChunkSchema(
            source_id=e.source_id,
            document_id=e.document_id,
            chunk_id=e.chunk_id,
            section=e.section,
            text=e.text,
            citation=e.citation,
            retrieval_method=e.score.retrieval_method,
            score_type=e.score.score_type.value if hasattr(e.score.score_type, "value") else str(e.score.score_type),
            score_value=e.score.score_value,
            ranking_position=e.score.ranking_position,
        )
        for e in turn.evidence
    ]
    
    tool_schemas = [
        ToolRecordSchema(
            tool_name=t.tool_name,
            arguments=t.arguments,
            result=t.result,
            origin=t.origin.value if hasattr(t.origin, "value") else str(t.origin),
            execution_time_ms=t.execution_time_ms,
        )
        for t in turn.tool_executions
    ]
    
    return ChatResponse(
        turn_id=turn.turn_id,
        session_id=turn.session_id,
        query=turn.user_input,
        answer=turn.canonical_answer,
        grounding_status=turn.grounding_status.value if hasattr(turn.grounding_status, "value") else str(turn.grounding_status),
        citations=turn.citations,
        invalid_citations=turn.invalid_citations,
        evidence=evidence_schemas,
        tools_used=tool_schemas,
        runtime_seconds=turn.runtime_seconds,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    session_service=Depends(get_session_service),
):
    """Retrieve canonical conversation history for a session."""
    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
        
    turn_schemas = [
        SessionTurnSchema(
            turn_id=t.turn_id,
            user_input=t.user_input,
            canonical_answer=t.canonical_answer,
            grounding_status=t.grounding_status.value if hasattr(t.grounding_status, "value") else str(t.grounding_status),
            citations=t.citations,
            created_at=t.created_at,
            runtime_seconds=t.runtime_seconds,
        )
        for t in session.turns
    ]
    
    return SessionResponse(
        session_id=session.session_id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        turn_count=len(session.turns),
        turns=turn_schemas,
    )
