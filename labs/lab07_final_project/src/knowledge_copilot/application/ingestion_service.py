"""Document ingestion lifecycle service with cross-store consistency and recovery."""
import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..domain.enums import DocumentStatus, EvidenceQuality
from ..domain.exceptions import DocumentNotFoundError, ExtractionError, IndexingError
from ..domain.models import Chunk, Document
from ..infrastructure.redis_store import RedisVectorAdapter, decode_redis
from ..infrastructure.sqlite_repo import SQLiteRepository
from ..observability.metrics import (
    RAG_INDEXED_CHUNKS,
    RAG_INDEXED_DOCUMENTS,
    RAG_INGESTION_FAILURES_TOTAL,
)
from ..rag.extractors import extract_text_from_bytes, validate_file_content
from ..rag.splitters import build_chunks_from_document

logger = logging.getLogger(__name__)


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
        self.corpus_version: int = 0
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
        if existing_doc and existing_chunks:
            return self.reindex_document(existing_doc.document_id, force=False)
        prior_doc = self.sqlite_repo.get_document(doc_id)
        if prior_doc and prior_doc.sha256 != sha256_hash:
            raise IndexingError("Document ID already belongs to different content; delete it explicitly before replacement.")

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

            # Structure-aware chunking with token bounding
            chunks = build_chunks_from_document(
                doc=doc,
                text=text,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                max_tokens=self.max_tokens,
                token_counter=self.embeddings.count_tokens if self.embeddings else None,
            )

            # Atomic indexing into Redis and SQLite
            if self.require_vector_store:
                if not self.redis_adapter.ping():
                    raise IndexingError("Vector store unreachable; indexing aborted to prevent false INDEXED state.")
                indexed_count = self.redis_adapter.index_chunks(chunks, self.embeddings)
                if len(chunks) > 0 and indexed_count != len(chunks):
                    raise IndexingError(f"Vector indexing incomplete: expected {len(chunks)} chunks, indexed {indexed_count}.")
            elif self.redis_adapter.ping():
                if self.redis_adapter.index_chunks(chunks, self.embeddings) != len(chunks):
                    raise IndexingError("Vector indexing incomplete.")

            self._cached_chunks[doc_id] = chunks
            try:
                self.sqlite_repo.save_chunks(chunks)
                doc.status = DocumentStatus.INDEXED
                doc.chunk_count = len(chunks)
                doc.updated_at = datetime.now(timezone.utc).isoformat()
                self.sqlite_repo.save_document(doc)
                self.corpus_version += 1
            except Exception as sql_exc:
                # Compensating rollback: remove Redis vectors if SQLite persistence fails
                if prior_doc is None:
                    self.redis_adapter.delete_document_chunks(doc_id)
                raise IndexingError("SQLite persistence failed during indexing.") from sql_exc

            self._reload_metrics()
            return doc

        except Exception as exc:
            doc.status = DocumentStatus.FAILED
            doc.error_message = f"{type(exc).__name__}: {str(exc)}"
            doc.updated_at = datetime.now(timezone.utc).isoformat()
            try:
                self.sqlite_repo.save_document(doc)
            except Exception as save_exc:
                logger.error("Could not persist ingestion failure (%s).", type(save_exc).__name__)

            # Atomic rollback: clear any partially added chunks
            if prior_doc is None:
                self.redis_adapter.delete_document_chunks(doc_id)
            self._cached_chunks.pop(doc_id, None)

            RAG_INGESTION_FAILURES_TOTAL.labels(error_type=type(exc).__name__).inc()
            self._reload_metrics()
            raise

    def reindex_document(self, document_id: str, *, force: bool = True) -> Document:
        """Rebuild vectors from persisted chunks without deleting valid indexed data."""
        doc = self.sqlite_repo.get_document(document_id)
        if doc is None:
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")
        chunks = self.sqlite_repo.get_chunks_for_document(document_id)
        if not chunks:
            raise IndexingError("Document has no persisted chunks to reindex.")
        try:
            indexed_count = self.redis_adapter.index_chunks(chunks, self.embeddings, force=force)
            if indexed_count != len(chunks):
                raise IndexingError(f"Vector indexing incomplete: expected {len(chunks)} chunks, indexed {indexed_count}.")
            doc.status = DocumentStatus.INDEXED
            doc.chunk_count = indexed_count
            doc.error_message = None
            doc.updated_at = datetime.now(timezone.utc).isoformat()
            self.sqlite_repo.save_document(doc)
        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError("Document reindexing failed; existing data was retained.") from exc
        self._cached_chunks[document_id] = chunks
        self.corpus_version += 1
        self._reload_metrics()
        return doc

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
        if success:
            self.corpus_version += 1
        self._reload_metrics()
        return success

    def reconcile_storage(self) -> Dict[str, Any]:
        """Verify cross-store consistency between SQLite and Redis and repair discrepancies."""
        report: Dict[str, Any] = {
            "reindexed_documents": [],
            "purged_redis_keys": 0,
            "failed_reconciliations": [],
            "status": "CONSISTENT",
        }
        if not self.redis_adapter.ping():
            report["status"] = "REDIS_UNREACHABLE"
            return report

        all_docs = self.sqlite_repo.list_documents()
        indexed_doc_ids = {d.document_id for d in all_docs if d.status == DocumentStatus.INDEXED}
        client = self.redis_adapter.get_client()

        # 1. Inspect Redis keys and purge orphans
        prefix = self.redis_adapter.key_prefix
        keys = list(client.scan_iter(match=f"{prefix}:*"))
        orphans = []
        for k in keys:
            try:
                k_type = decode_redis(client.type(k))
                if k_type != "hash":
                    continue
                doc_val = client.hget(k, "document_id") or client.hget(k, "source_id")
                if doc_val:
                    d_id = decode_redis(doc_val)
                    if d_id not in indexed_doc_ids:
                        orphans.append(k)
                else:
                    orphans.append(k)
            except Exception:
                continue

        if orphans:
            client.delete(*orphans)
            report["purged_redis_keys"] = len(orphans)

        # 2. Check each INDEXED document has corresponding vectors
        for doc in all_docs:
            if doc.status == DocumentStatus.INDEXED:
                chunks = self.sqlite_repo.get_chunks_for_document(doc.document_id)
                missing_any = False
                for c in chunks:
                    k = f"{prefix}:{c.chunk_id}"
                    if not client.exists(k):
                        missing_any = True
                        break
                if missing_any and chunks:
                    try:
                        self.redis_adapter.index_chunks(chunks, self.embeddings)
                        report["reindexed_documents"].append(doc.document_id)
                    except Exception as exc:
                        report["failed_reconciliations"].append({
                            "document_id": doc.document_id,
                            "error": str(exc),
                        })
            elif doc.status in (DocumentStatus.PROCESSING, DocumentStatus.FAILED):
                self.redis_adapter.delete_document_chunks(doc.document_id)

        if report["failed_reconciliations"]:
            report["status"] = "REPAIR_INCOMPLETE"
        elif report["reindexed_documents"] or report["purged_redis_keys"] > 0:
            report["status"] = "REPAIRED"

        self._reload_metrics()
        return report

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
