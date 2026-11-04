"""RAG and Agent Query Coordination Service."""
import contextvars
import re
import time
from pathlib import Path
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

ACTIVE_RETRIEVAL_MODE: contextvars.ContextVar[Optional[RetrievalMode]] = contextvars.ContextVar(
    "active_retrieval_mode", default=None
)


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
        active_mode = mode or ACTIVE_RETRIEVAL_MODE.get() or self.default_retrieval_mode
        start_time = time.perf_counter()

        # Dynamic document routing based on indexed catalog
        targets: List[str] = []
        if not document_id:
            referenced_docs: List[str] = []
            for doc in self.ingestion_service.list_documents():
                stem = Path(doc.filename).stem.casefold()
                clean_stem = re.sub(r"[_-]", " ", stem)
                pattern = r"\b" + re.escape(clean_stem) + r"\b"
                if stem in query.casefold() or re.search(pattern, query.casefold()):
                    referenced_docs.append(doc.document_id)
                    continue
                lab_match = re.search(r"(?:lab|лабораторн\w*)[_\s-]*№?\s*0?([1-9])", stem)
                if lab_match:
                    num = lab_match.group(1)
                    if re.search(rf"\b(?:lab|лабораторн\w*(?:\s*робот\w*)?)\s*№?\s*0?{num}\b", query.casefold()):
                        referenced_docs.append(doc.document_id)
                        continue
                adr_match = re.search(r"adr[_\s-]*0*([1-9]+)", stem)
                if adr_match:
                    num = adr_match.group(1)
                    if re.search(rf"\b(?:adr[_\s-]*0*{num}|lab\s*0?7|специфікац\w*)\b", query.casefold()):
                        referenced_docs.append(doc.document_id)
                        continue
                if "postmortem" in stem or "incident" in stem:
                    if re.search(r"\b(?:postmortem|post-mortem|інцидент\w*|incident|oom)\b", query.casefold()):
                        referenced_docs.append(doc.document_id)
                        continue
                if "runbook" in stem:
                    if re.search(r"\b(?:runbook|експлуатаці\w*|operating procedure|knowledge copilot|copilot)\b", query.casefold()):
                        referenced_docs.append(doc.document_id)
                        continue
                if "api" in stem:
                    if re.search(r"\b(?:api|ендпоінт\w*|endpoints?)\b", query.casefold()):
                        referenced_docs.append(doc.document_id)
                        continue

            if len(referenced_docs) == 1:
                document_id = referenced_docs[0]
            elif len(referenced_docs) > 1:
                targets = referenced_docs

        evidence_chunks: List[EvidenceChunk] = []

        if active_mode == RetrievalMode.DENSE:
            if targets:
                dense_results = []
                per_doc_k = max(2, (k // len(targets)) + 1)
                for t_id in targets:
                    dense_results.extend(self.redis_adapter.dense_search(
                        query=query, k=per_doc_k, document_id=t_id, filter_quoted_outputs=filter_quoted_outputs
                    ))
            else:
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
            if targets:
                lex_results = []
                per_doc_k = max(2, (k // len(targets)) + 1)
                for t_id in targets:
                    lex_results.extend(self.bm25.search(
                        query=query, k=per_doc_k, document_id=t_id, filter_quoted_outputs=filter_quoted_outputs
                    ))
            else:
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
            if targets:
                per_doc_quota = max(2, (k // len(targets)) + 1)
                per_target_chunks: Dict[str, List[EvidenceChunk]] = {}
                for t_id in targets:
                    dense_t = self.redis_adapter.dense_search(
                        query=query, k=per_doc_quota * 4, document_id=t_id, filter_quoted_outputs=filter_quoted_outputs
                    )
                    lex_t = self.bm25.search(
                        query=query, k=per_doc_quota * 4, document_id=t_id, filter_quoted_outputs=filter_quoted_outputs
                    )
                    fused_t = reciprocal_rank_fusion(
                        dense_results=dense_t,
                        lexical_results=lex_t,
                        k=per_doc_quota * 2,
                        dense_weight=1.5,
                        lexical_weight=1.0,
                    )
                    per_target_chunks[t_id] = fused_t

                interleaved: List[EvidenceChunk] = []
                for idx in range(per_doc_quota * 2):
                    for t_id in targets:
                        if idx < len(per_target_chunks[t_id]):
                            c = per_target_chunks[t_id][idx]
                            if c.chunk.chunk_id not in {x.chunk.chunk_id for x in interleaved}:
                                interleaved.append(c)
                        if len(interleaved) >= k:
                            break
                    if len(interleaved) >= k:
                        break
                evidence_chunks = interleaved[:k]
            else:
                dense_candidates = self.redis_adapter.dense_search(
                    query=query,
                    k=k * 4,
                    document_id=document_id,
                    filter_quoted_outputs=filter_quoted_outputs,
                )
                lex_candidates = self.bm25.search(
                    query=query,
                    k=k * 4,
                    document_id=document_id,
                    filter_quoted_outputs=filter_quoted_outputs,
                )
                evidence_chunks = reciprocal_rank_fusion(
                    dense_results=dense_candidates,
                    lexical_results=lex_candidates,
                    k=k,
                    dense_weight=1.5,
                    lexical_weight=1.0,
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

        token = ACTIVE_RETRIEVAL_MODE.set(retrieval_mode)
        try:
            turn = await self.orchestrator.execute_turn(
                user_input=query,
                session_id=session_id,
                history_messages=history_messages,
                retrieval_mode=retrieval_mode,
            )
        finally:
            ACTIVE_RETRIEVAL_MODE.reset(token)

        # Transactional SQLite canonical synchronization
        self.sqlite_repo.save_canonical_turn(turn)

        # Observability metrics
        duration = time.perf_counter() - start_req
        status_label = "SUCCESS" if turn.grounding_status != GroundingStatus.EXECUTION_FAILED else "FAILED"
        RAG_REQUESTS_TOTAL.labels(status=status_label, grounding_status=turn.grounding_status.value).inc()
        RAG_REQUEST_DURATION_SECONDS.observe(duration)

        return turn
