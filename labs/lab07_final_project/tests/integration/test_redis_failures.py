"""Tests for Redis failure handling, atomic rollbacks, idempotency, and deletion consistency (Gate B3)."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from labs.lab07_final_project.src.knowledge_copilot.domain.enums import DocumentStatus, RetrievalMode
from labs.lab07_final_project.src.knowledge_copilot.domain.exceptions import DocumentNotFoundError, IndexingError
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, Document
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository
from labs.lab07_final_project.src.knowledge_copilot.application.ingestion_service import IngestionService
from labs.lab07_final_project.src.knowledge_copilot.application.query_service import QueryService


class MockFastEmbeddings:
    """Fast deterministic mock embeddings for offline integration tests."""
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts):
        return [[0.05] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.05] * 384

    def probe_dimension(self) -> int:
        return 384


@pytest.fixture
def sqlite_repo(tmp_path: Path) -> SQLiteRepository:
    return SQLiteRepository(tmp_path / "redis_fail_test.sqlite")


@pytest.fixture
def live_or_mock_redis():
    """Use live Redis Stack on port 6380 if available, else a mock."""
    adapter = RedisVectorAdapter(redis_url="redis://localhost:6380", index_name="test_redis_fail_idx")
    if adapter.ping():
        # Clean test index keys before test
        client = adapter.get_client()
        keys = list(client.scan_iter("lab07:kb:test_*"))
        if keys:
            client.delete(*keys)
        return adapter
    # If live Redis is not listening, provide an offline mock
    mock = MagicMock(spec=RedisVectorAdapter)
    mock.ping.return_value = False
    return mock


# ============================================================================
# 1. Redis Failure Before Indexing (No False INDEXED Status)
# ============================================================================

def test_redis_offline_prevents_false_indexed_status(tmp_path: Path, sqlite_repo: SQLiteRepository):
    """When Redis is completely unreachable during strict indexing, document must NOT be left INDEXED."""
    offline_redis = RedisVectorAdapter(redis_url="redis://localhost:9999", index_name="unreachable_idx")
    assert not offline_redis.ping()

    svc = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=offline_redis,
        embeddings_model=MockFastEmbeddings(),
        require_vector_store=True,
    )

    content = b"# Document Title\nImportant technical content."
    with pytest.raises(IndexingError, match="Vector store unreachable"):
        svc.ingest_file("offline_test.md", content, document_id="doc_offline_1")

    stored = sqlite_repo.get_document("doc_offline_1")
    assert stored is not None
    assert stored.status == DocumentStatus.FAILED, f"Expected FAILED status, got {stored.status}"
    assert "Vector store unreachable" in (stored.error_message or "")


# ============================================================================
# 2. Redis Failure During Indexing Triggers Atomic Rollback
# ============================================================================

def test_redis_failure_during_indexing_rolls_back(tmp_path: Path, sqlite_repo: SQLiteRepository):
    """If an exception occurs during chunk indexing, status must be FAILED, not INDEXED."""
    failing_redis = MagicMock(spec=RedisVectorAdapter)
    failing_redis.ping.return_value = True
    failing_redis.index_chunks.side_effect = RuntimeError("Redis connection dropped midway through transaction")

    svc = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=failing_redis,
        embeddings_model=MockFastEmbeddings(),
    )

    with pytest.raises(RuntimeError, match="Redis connection dropped"):
        svc.ingest_file(
            filename="crash_test.md",
            content=b"# Header\nWill crash during vector index.",
            document_id="doc_crash_01",
        )

    # Verification: SQLite record must be FAILED, NEVER left falsely INDEXED
    doc_record = sqlite_repo.get_document("doc_crash_01")
    assert doc_record is not None
    assert doc_record.status == DocumentStatus.FAILED
    assert "RuntimeError" in (doc_record.error_message or "")
    # Rollback must have called delete_document_chunks
    failing_redis.delete_document_chunks.assert_called_with("doc_crash_01")


# ============================================================================
# 3. No Duplicate Vectors After Recovery / Reindexing
# ============================================================================

def test_no_duplicate_vectors_after_repeated_ingestion(sqlite_repo: SQLiteRepository, live_or_mock_redis):
    """Re-ingesting an identical document must return the existing document and not duplicate chunks."""
    svc = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=live_or_mock_redis,
        embeddings_model=MockFastEmbeddings(),
    )

    content = b"# Idempotent Spec\nDeterministic content for hashing."
    doc1 = svc.ingest_file("idemp.md", content, document_id="doc_idemp_1")
    initial_chunks = len(sqlite_repo.get_chunks_for_document(doc1.document_id))

    # Second ingestion of identical content
    doc2 = svc.ingest_file("idemp.md", content, document_id="doc_idemp_1")
    repeat_chunks = len(sqlite_repo.get_chunks_for_document(doc2.document_id))

    assert doc1.document_id == doc2.document_id
    assert doc1.sha256 == doc2.sha256
    assert initial_chunks == repeat_chunks, "Chunk count must not increase on duplicate ingestion"


# ============================================================================
# 4. Deleted Document Absent from Subsequent Retrieval
# ============================================================================

def test_deleted_document_absent_from_subsequent_retrieval(sqlite_repo: SQLiteRepository, live_or_mock_redis):
    """After successful deletion, document chunks must never appear in subsequent retrieval."""
    svc = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=live_or_mock_redis,
        embeddings_model=MockFastEmbeddings(),
    )
    query_svc = QueryService(
        ingestion_service=svc,
        redis_adapter=live_or_mock_redis,
        sqlite_repo=sqlite_repo,
    )

    content = b"# Secret Deletion Test\nUniqueEphemeralKeyword999 that should be erased."
    doc = svc.ingest_file("secret.md", content, document_id="doc_secret_delete")

    # Ingested and searchable
    results_before = query_svc.search("UniqueEphemeralKeyword999", k=5, mode=RetrievalMode.LEXICAL)
    assert len(results_before) >= 1
    assert any(c.chunk.document_id == "doc_secret_delete" for c in results_before)

    # Perform atomic deletion
    deleted = svc.delete_document("doc_secret_delete")
    assert deleted is True

    # Subsequent retrieval must NOT contain deleted document
    results_after = query_svc.search("UniqueEphemeralKeyword999", k=5, mode=RetrievalMode.LEXICAL)
    assert not any(c.chunk.document_id == "doc_secret_delete" for c in results_after)
    assert sqlite_repo.get_document("doc_secret_delete") is None
    assert len(sqlite_repo.get_chunks_for_document("doc_secret_delete")) == 0


# ============================================================================
# 5. Non-Existent Document Deletion Raises Explicit Domain Exception
# ============================================================================

def test_nonexistent_document_deletion_raises_not_found(sqlite_repo: SQLiteRepository, live_or_mock_redis):
    """Deleting a non-existent document must not silently succeed."""
    svc = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=live_or_mock_redis,
        embeddings_model=MockFastEmbeddings(),
    )

    with pytest.raises(DocumentNotFoundError, match="Document 'ghost_doc' not found"):
        svc.delete_document("ghost_doc")
