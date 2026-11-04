"""Document ingestion lifecycle service."""
import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..domain.enums import DocumentStatus, EvidenceQuality
from ..domain.exceptions import DocumentNotFoundError, ExtractionError, IndexingError
from ..domain.models import Chunk, Document
from ..infrastructure.redis_store import RedisVectorAdapter
from ..infrastructure.sqlite_repo import SQLiteRepository
from ..observability.metrics import (
    RAG_INDEXED_CHUNKS,
    RAG_INDEXED_DOCUMENTS,
    RAG_INGESTION_FAILURES_TOTAL,
)
from ..rag.extractors import extract_text_from_bytes, validate_file_content
from ..rag.splitters import build_chunks_from_document


class IngestionService:
    def __init__(
        self,
        sqlite_repo: SQLiteRepository,
        redis_adapter: RedisVectorAdapter,
        embeddings_model,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        max_tokens: int = 240,
        require_vector_store: bool = True,
    ):
        self.sqlite_repo = sqlite_repo
        self.redis_adapter = redis_adapter
        self.embeddings = embeddings_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.require_vector_store = require_vector_store
        self._cached_chunks: Dict[str, List[Chunk]] = {}
        self._reload_metrics()

    def _reload_metrics(self) -> None:
        docs = self.sqlite_repo.list_documents()
        indexed_docs = [d for d in docs if d.status == DocumentStatus.INDEXED]
        RAG_INDEXED_DOCUMENTS.set(len(indexed_docs))
        total_chunks = sum(d.chunk_count for d in indexed_docs)
        RAG_INDEXED_CHUNKS.set(total_chunks)

    def ingest_file(
        self,
        filename: str,
        content: bytes,
        collection: str = "default",
        document_id: Optional[str] = None,
    ) -> Document:
        """Execute full document lifecycle: validation -> parsing -> chunking -> atomic indexing."""
        clean_name, sha256_hash, size_bytes = validate_file_content(content, filename)
        doc_id = document_id or f"doc_{sha256_hash[:12]}"
        
        # Idempotency check: prevent duplicate vectors for identical content
        existing_doc = self.sqlite_repo.get_document_by_sha256(sha256_hash, collection=collection)
        existing_chunks = self.sqlite_repo.get_chunks_for_document(existing_doc.document_id) if existing_doc else []
        if existing_doc and existing_doc.status == DocumentStatus.INDEXED and existing_chunks:
            self._cached_chunks[existing_doc.document_id] = existing_chunks
            return existing_doc

        content_type = Path(clean_name).suffix.lstrip(".").lower()
        now = datetime.now(timezone.utc).isoformat()
        
        doc = Document(
            document_id=doc_id,
            filename=clean_name,
            content_type=content_type,
            sha256=sha256_hash,
            size_bytes=size_bytes,
            status=DocumentStatus.PROCESSING,
            chunk_count=0,
            collection=collection,
            created_at=now,
            updated_at=now,
        )
        self.sqlite_repo.save_document(doc)

        try:
            # Extract text (raises ExtractionError for scanned PDF without text layer)
            text = extract_text_from_bytes(content, clean_name)
            
            # Structure-aware chunking with token bounding (0 truncated inputs)
            chunks = build_chunks_from_document(
                doc=doc,
                text=text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                max_tokens=self.max_tokens,
                token_counter=self.embeddings.count_tokens if self.embeddings else None,
            )
            
            # Atomic indexing into Redis and SQLite
            self.redis_adapter.delete_document_chunks(doc_id)
            if self.require_vector_store:
                if not self.redis_adapter.ping():
                    r_url = getattr(self.redis_adapter, "redis_url", "configured endpoint")
                    raise IndexingError(f"Vector store unreachable at {r_url}; indexing aborted to prevent false INDEXED state.")
                self.redis_adapter.index_chunks(chunks, self.embeddings)
            elif self.redis_adapter.ping():
                self.redis_adapter.index_chunks(chunks, self.embeddings)
                
            self._cached_chunks[doc_id] = chunks
            self.sqlite_repo.save_chunks(chunks)
            
            doc.status = DocumentStatus.INDEXED
            doc.chunk_count = len(chunks)
            doc.updated_at = datetime.now(timezone.utc).isoformat()
            self.sqlite_repo.save_document(doc)
            self._reload_metrics()
            return doc
            
        except Exception as exc:
            doc.status = DocumentStatus.FAILED
            doc.error_message = f"{type(exc).__name__}: {str(exc)}"
            doc.updated_at = datetime.now(timezone.utc).isoformat()
            self.sqlite_repo.save_document(doc)
            
            # Atomic rollback: clear any partially added chunks
            self.redis_adapter.delete_document_chunks(doc_id)
            self._cached_chunks.pop(doc_id, None)
            
            RAG_INGESTION_FAILURES_TOTAL.labels(error_type=type(exc).__name__).inc()
            self._reload_metrics()
            raise

    def delete_document(self, document_id: str) -> bool:
        """Delete document from SQLite and remove all associated vectors from Redis."""
        doc = self.sqlite_repo.get_document(document_id)
        if not doc:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")
            
        if self.require_vector_store and not self.redis_adapter.ping():
            r_url = getattr(self.redis_adapter, "redis_url", "configured endpoint")
            raise IndexingError(f"Vector store unreachable at {r_url}; cannot reliably delete vectors for '{document_id}'.")

        # Remove from Redis and SQLite
        self.redis_adapter.delete_document_chunks(document_id)
        self.sqlite_repo.delete_chunks_for_document(document_id)
        self._cached_chunks.pop(document_id, None)
        
        # Remove from SQLite
        success = self.sqlite_repo.delete_document(document_id)
        self._reload_metrics()
        return success

    def list_documents(self, collection: Optional[str] = None) -> List[Document]:
        return self.sqlite_repo.list_documents(collection=collection)

    def get_document(self, document_id: str) -> Optional[Document]:
        return self.sqlite_repo.get_document(document_id)

    def get_all_indexed_chunks(self) -> List[Chunk]:
        """Aggregate chunks for lexical retrieval from SQLite or memory."""
        stored = self.sqlite_repo.get_all_chunks()
        if stored:
            return stored
        all_chunks: List[Chunk] = []
        for chunks in self._cached_chunks.values():
            all_chunks.extend(chunks)
        return all_chunks
