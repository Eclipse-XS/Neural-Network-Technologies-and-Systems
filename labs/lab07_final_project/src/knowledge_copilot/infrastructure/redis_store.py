"""Redis Stack vector store and index management."""
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from redis import Redis

from ..domain.enums import EvidenceQuality
from ..domain.models import Chunk


def decode_redis(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (list, tuple)):
        return [decode_redis(v) for v in value]
    if isinstance(value, dict):
        return {decode_redis(k): decode_redis(v) for k, v in value.items()}
    return value


class RedisVectorAdapter:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6380",
        index_name: str = "lab07_course_kb_idx",
        key_prefix: str = "lab07:kb",
    ):
        self.redis_url = redis_url
        self.index_name = index_name
        self.key_prefix = key_prefix
        self._client: Optional[Redis] = None
        self._store = None
        self._schema = [
            {"name": "source_id", "type": "tag"},
            {"name": "document_id", "type": "tag"},
            {"name": "filename", "type": "tag"},
            {"name": "section", "type": "tag"},
            {"name": "chunk_id", "type": "tag"},
            {"name": "collection", "type": "tag"},
            {"name": "evidence_quality", "type": "tag"},
        ]

    def get_client(self) -> Redis:
        if self._client is None:
            self._client = Redis.from_url(
                self.redis_url,
                socket_connect_timeout=3,
                socket_timeout=10,
                decode_responses=False,
            )
        return self._client

    def ping(self) -> bool:
        try:
            return bool(self.get_client().ping())
        except Exception:
            return False

    def initialize_store(self, embeddings, chunks: Optional[List[Chunk]] = None) -> None:
        from langchain_redis import RedisConfig, RedisVectorStore
        
        client = self.get_client()
        existing = decode_redis(client.execute_command("FT._LIST"))
        dim = embeddings.probe_dimension()
        
        config = RedisConfig(
            index_name=self.index_name,
            key_prefix=self.key_prefix,
            redis_client=client,
            embedding_dimensions=dim,
            distance_metric="COSINE",
            indexing_algorithm="FLAT",
            vector_datatype="FLOAT32",
            storage_type="hash",
            metadata_schema=self._schema,
            legacy_key_format=False,
        )
        self._store = RedisVectorStore(embeddings, config=config)
        
        if chunks:
            self.index_chunks(chunks, embeddings)

    def index_chunks(self, chunks: List[Chunk], embeddings) -> int:
        """Atomically index chunks into Redis without duplicating vectors."""
        if not chunks or not self.ping():
            return 0
        if self._store is None:
            self.initialize_store(embeddings)
        from langchain_core.documents import Document as LCDocument

        # Check existing chunk keys
        client = self.get_client()
        missing_chunks: List[Chunk] = []
        for c in chunks:
            key = f"{self.key_prefix}:{c.chunk_id}"
            if not client.exists(key):
                missing_chunks.append(c)

        if not missing_chunks:
            return 0

        lc_docs = []
        for c in missing_chunks:
            meta = {
                "source_id": c.document_id,
                "document_id": c.document_id,
                "filename": c.metadata.get("filename", f"{c.document_id}.md"),
                "section": c.section,
                "chunk_id": c.chunk_id,
                "collection": c.metadata.get("collection", "default"),
                "evidence_quality": c.evidence_quality.value if isinstance(c.evidence_quality, EvidenceQuality) else str(c.evidence_quality),
            }
            if "lab_number" in c.metadata:
                meta["lab_number"] = str(c.metadata["lab_number"])
            lc_docs.append(LCDocument(page_content=c.text, metadata=meta))

        # Add to vector store
        ids = [c.chunk_id for c in missing_chunks]
        self._store.add_documents(lc_docs, ids=ids)
        return len(missing_chunks)

    def delete_document_chunks(self, document_id: str) -> int:
        """Remove all indexed chunks belonging to a document so they no longer appear in search."""
        if not self.ping():
            return 0
        client = self.get_client()
        keys = list(client.scan_iter(match=f"{self.key_prefix}:*"))
        to_delete = []
        for k in keys:
            try:
                k_type = decode_redis(client.type(k))
                if k_type != "hash":
                    continue
                doc_val = client.hget(k, "document_id") or client.hget(k, "source_id")
                if doc_val and decode_redis(doc_val) == document_id:
                    to_delete.append(k)
            except Exception:
                continue

        if to_delete:
            deleted_count = client.delete(*to_delete)
            return deleted_count
        return 0

    def dense_search(
        self,
        query: str,
        k: int = 5,
        document_id: Optional[str] = None,
        filter_quoted_outputs: bool = True,
    ) -> List[Tuple[Chunk, float]]:
        """Return (Chunk, cosine_distance) sorted by distance asc (lower is closer)."""
        if not self.ping() or self._store is None:
            return []
        from redisvl.query.filter import Tag

        expr = None
        if document_id:
            expr = (Tag("document_id") == document_id) | (Tag("source_id") == document_id)
        if filter_quoted_outputs:
            quality_filter = (Tag("evidence_quality") == "canonical_report") | (Tag("evidence_quality") == "user_document")
            expr = expr & quality_filter if expr is not None else quality_filter

        raw_results = self._store.similarity_search_with_score(query, k=k, filter=expr)
        
        results: List[Tuple[Chunk, float]] = []
        for doc, distance in raw_results:
            meta = dict(doc.metadata)
            doc_id = meta.get("document_id") or meta.get("source_id", "unknown")
            chunk_id = meta.get("chunk_id", "unknown")
            section = meta.get("section", "General")
            quality = EvidenceQuality(meta.get("evidence_quality", "user_document"))
            
            chunk = Chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                ordinal=meta.get("ordinal", 0),
                text=doc.page_content,
                section=section,
                token_count=len(doc.page_content.split()),
                sha256=hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest(),
                evidence_quality=quality,
                metadata=meta,
            )
            results.append((chunk, float(distance)))

        results.sort(key=lambda item: (item[1], item[0].chunk_id))
        return results[:k]
