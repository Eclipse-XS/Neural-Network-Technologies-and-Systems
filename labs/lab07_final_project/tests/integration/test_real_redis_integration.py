"""Mandatory Gate P0-10: Independent Real Redis Integration Test Suite.
Requires an actual live Redis Stack instance on port 6380.
Strictly prohibits MagicMock / Mock adapters for Redis.
Fails explicitly if Redis is unreachable on localhost:6380.
"""
import uuid
import numpy as np
import pytest
import redis
from langchain_core.embeddings import Embeddings

from labs.lab07_final_project.src.knowledge_copilot.domain.enums import EvidenceQuality
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter


class RealSyntheticEmbeddings(Embeddings):
    """Deterministic token-based 384-dimensional test embedding adapter for real vector search."""
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def probe_dimension(self) -> int:
        return self.dimension

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _text_to_vec(self, text: str) -> list:
        words = text.lower().split()
        v = np.zeros(self.dimension, dtype=np.float32)
        for w in words:
            seed = sum(ord(c) * (i + 1) for i, c in enumerate(w)) % 100000
            rng = np.random.RandomState(seed)
            v += rng.randn(self.dimension).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v.tolist()

    def embed_documents(self, texts):
        return [self._text_to_vec(t) for t in texts]

    def embed_query(self, text: str):
        return self._text_to_vec(text)


@pytest.fixture(scope="module")
def live_redis_adapter():
    import os
    candidate_urls = [os.environ.get("REDIS_URL"), "redis://localhost:6380", "redis://localhost:6379"]
    raw_client = None
    redis_url = None
    last_exc = None
    for url in candidate_urls:
        if not url:
            continue
        try:
            c = redis.from_url(url, socket_connect_timeout=2)
            if c.ping():
                raw_client = c
                redis_url = url
                break
        except Exception as exc:
            last_exc = exc
    if raw_client is None:
        pytest.fail(f"Mandatory Gate P0-10: Real Redis Stack on ports 6380/6379 is unreachable: {last_exc}")

    uid = uuid.uuid4().hex[:8]
    test_idx = f"real_redis_test_idx_{uid}"
    key_prefix = f"real_test_pfx_{uid}"
    adapter = RedisVectorAdapter(redis_url=redis_url, index_name=test_idx, key_prefix=key_prefix)
    embeddings = RealSyntheticEmbeddings(384)
    adapter.initialize_store(embeddings)

    yield adapter, embeddings

    # Teardown: drop test index and delete indexed documents
    try:
        raw_client.execute_command("FT.DROPINDEX", test_idx, "DD")
    except Exception:
        pass


def test_real_redis_ping_and_index_creation(live_redis_adapter):
    """Verify live Redis instance is healthy and vector index is initialized without mocks."""
    adapter, embeddings = live_redis_adapter
    assert adapter.ping() is True, "Live Redis on port 6380 must respond True to ping()"


def test_real_redis_vector_indexing_and_similarity_retrieval(live_redis_adapter):
    """Verify live vector indexing, dense search, and score computation on real Redis."""
    adapter, embeddings = live_redis_adapter
    doc_id = f"real_doc_{uuid.uuid4().hex[:8]}"

    chunk1 = Chunk(
        chunk_id="chunk_real_01",
        document_id=doc_id,
        ordinal=0,
        text="FastAPI server handles high concurrency using async request handlers.",
        section="Architecture",
        token_count=12,
        sha256="sha_real_01",
        evidence_quality=EvidenceQuality.USER_DOCUMENT,
    )
    chunk2 = Chunk(
        chunk_id="chunk_real_02",
        document_id=doc_id,
        ordinal=1,
        text="Redis Stack provides vector search capabilities with HNSW and Flat indices.",
        section="Storage",
        token_count=13,
        sha256="sha_real_02",
        evidence_quality=EvidenceQuality.USER_DOCUMENT,
    )

    adapter.index_chunks([chunk1, chunk2], embeddings)

    # Search for Redis Stack content
    results = adapter.dense_search(
        query="Redis Stack vector search capabilities",
        k=2,
        document_id=doc_id,
    )

    assert len(results) >= 1, "Expected at least 1 retrieved result from real Redis"
    top_chunk, top_dist = results[0]
    assert top_chunk.chunk_id == "chunk_real_02", f"Expected chunk_real_02 as top match, got {top_chunk.chunk_id}"
    assert top_chunk.document_id == doc_id
    assert isinstance(top_dist, float)
    assert top_dist >= 0.0, "Cosine distance must be non-negative"


def test_real_redis_document_deletion_removes_vectors(live_redis_adapter):
    """Verify that deleting document chunks on real Redis permanently removes keys."""
    adapter, embeddings = live_redis_adapter
    doc_id = f"delete_doc_{uuid.uuid4().hex[:8]}"

    chunk = Chunk(
        chunk_id="chunk_del_01",
        document_id=doc_id,
        ordinal=0,
        text="Temporary configuration document to be purged from vector store.",
        section="Config",
        token_count=10,
        sha256="sha_del_01",
        evidence_quality=EvidenceQuality.USER_DOCUMENT,
    )

    adapter.index_chunks([chunk], embeddings)

    # Confirm present in real Redis
    pre_results = adapter.dense_search(query="Temporary configuration", k=5, document_id=doc_id)
    assert len(pre_results) == 1

    # Delete
    deleted_count = adapter.delete_document_chunks(doc_id)
    assert deleted_count == 1

    # Confirm absent in real Redis
    post_results = adapter.dense_search(query="Temporary configuration", k=5, document_id=doc_id)
    assert len(post_results) == 0, "Deleted document must return 0 results from real Redis"
