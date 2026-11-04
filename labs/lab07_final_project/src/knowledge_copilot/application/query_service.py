"""RAG and Agent Query Coordination Service."""
import time
from typing import Any, Dict, List, Optional

from ..domain.enums import GroundingStatus, RetrievalMode, ScoreType
from ..domain.models import EvidenceChunk, Score, Turn
from ..infrastructure.redis_store import RedisVectorAdapter
from ..infrastructure.sqlite_repo import SQLiteRepository
from ..observability.metrics import (
    RAG_REQUEST_DURATION_SECONDS,
    RAG_REQUESTS_TOTAL,
    RAG_RETRIEVAL_DURATION_SECONDS,
    RAG_RETRIEVAL_EMPTY_TOTAL,
)
from ..rag.hybrid import reciprocal_rank_fusion
from ..rag.lexical import BM25Retriever
from .ingestion_service import IngestionService


class QueryService:
    def __init__(
        self,
        ingestion_service: IngestionService,
        redis_adapter: RedisVectorAdapter,
        sqlite_repo: SQLiteRepository,
        orchestrator=None,
        default_retrieval_mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = 5,
    ):
        self.ingestion_service = ingestion_service
        self.redis_adapter = redis_adapter
        self.sqlite_repo = sqlite_repo
        self.orchestrator = orchestrator
        self.default_retrieval_mode = default_retrieval_mode
        self.top_k = top_k
        self.bm25 = BM25Retriever()
        self._refresh_lexical_index()

    def _refresh_lexical_index(self) -> None:
        chunks = self.ingestion_service.get_all_indexed_chunks()
        if chunks:
            self.bm25.index(chunks)

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        document_id: Optional[str] = None,
        mode: Optional[RetrievalMode] = None,
        filter_quoted_outputs: bool = True,
    ) -> List[EvidenceChunk]:
        """Perform multi-mode retrieval with explicit score semantics (DEFECT-06)."""
        k = k or self.top_k
        active_mode = mode or self.default_retrieval_mode
        start_time = time.perf_counter()

        evidence_chunks: List[EvidenceChunk] = []

        if active_mode == RetrievalMode.DENSE:
            dense_results = self.redis_adapter.dense_search(
                query=query,
                k=k,
                document_id=document_id,
                filter_quoted_outputs=filter_quoted_outputs,
            )
            for rank, (chunk, distance) in enumerate(dense_results, 1):
                score = Score(
                    retrieval_method="dense",
                    score_type=ScoreType.COSINE_DISTANCE,
                    score_value=distance,
                    ranking_position=rank,
                )
                evidence_chunks.append(EvidenceChunk(chunk=chunk, score=score))

        elif active_mode == RetrievalMode.LEXICAL:
            self._refresh_lexical_index()
            lex_results = self.bm25.search(
                query=query,
                k=k,
                document_id=document_id,
                filter_quoted_outputs=filter_quoted_outputs,
            )
            for rank, (chunk, bm25_score) in enumerate(lex_results, 1):
                score = Score(
                    retrieval_method="lexical",
                    score_type=ScoreType.BM25_SCORE,
                    score_value=bm25_score,
                    ranking_position=rank,
                )
                evidence_chunks.append(EvidenceChunk(chunk=chunk, score=score))

        elif active_mode == RetrievalMode.HYBRID:
            self._refresh_lexical_index()
            # Fetch candidates from both channels
            dense_candidates = self.redis_adapter.dense_search(
                query=query,
                k=k * 2,
                document_id=document_id,
                filter_quoted_outputs=filter_quoted_outputs,
            )
            lex_candidates = self.bm25.search(
                query=query,
                k=k * 2,
                document_id=document_id,
                filter_quoted_outputs=filter_quoted_outputs,
            )
            evidence_chunks = reciprocal_rank_fusion(
                dense_results=dense_candidates,
                lexical_results=lex_candidates,
                k=k,
            )

        duration = time.perf_counter() - start_time
        RAG_RETRIEVAL_DURATION_SECONDS.labels(mode=active_mode.value).observe(duration)
        if not evidence_chunks:
            RAG_RETRIEVAL_EMPTY_TOTAL.inc()

        return evidence_chunks

    def list_sources(self) -> List[Dict[str, Any]]:
        docs = self.ingestion_service.list_documents()
        return [
            {
                "source_id": d.document_id,
                "document_id": d.document_id,
                "filename": d.filename,
                "chunk_count": d.chunk_count,
                "status": d.status.value,
                "collection": d.collection,
            }
            for d in docs
        ]

    async def execute_query(
        self,
        query: str,
        session_id: str,
        retrieval_mode: Optional[RetrievalMode] = None,
    ) -> Turn:
        """Execute conversational or RAG query, synchronize canonical SQLite session (DEFECT-01)."""
        start_req = time.perf_counter()
        session = self.sqlite_repo.get_session(session_id)
        if not session:
            session = self.sqlite_repo.create_session(session_id)

        history_messages = []
        for t in session.turns:
            history_messages.append({"role": "user", "content": t.user_input})
            history_messages.append({"role": "assistant", "content": t.canonical_answer})

        if not self.orchestrator:
            raise RuntimeError("AgentOrchestrator is not configured on QueryService.")

        turn = await self.orchestrator.execute_turn(
            user_input=query,
            session_id=session_id,
            history_messages=history_messages,
        )

        # Transactional SQLite canonical synchronization
        self.sqlite_repo.save_canonical_turn(turn)

        # Observability metrics
        duration = time.perf_counter() - start_req
        status_label = "SUCCESS" if turn.grounding_status != GroundingStatus.EXECUTION_FAILED else "FAILED"
        RAG_REQUESTS_TOTAL.labels(status=status_label, grounding_status=turn.grounding_status.value).inc()
        RAG_REQUEST_DURATION_SECONDS.observe(duration)

        return turn
