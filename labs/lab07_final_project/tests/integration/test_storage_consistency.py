"""Integration tests for cross-store consistency, compensating rollbacks, and storage reconciliation."""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import DocumentStatus
from labs.lab07_final_project.src.knowledge_copilot.domain.exceptions import IndexingError
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, Document
from labs.lab07_final_project.src.knowledge_copilot.application.ingestion_service import IngestionService
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter


@pytest.fixture
def temp_repo(tmp_path):
    db_path = tmp_path / "test_consistency.db"
    return SQLiteRepository(db_path)


@pytest.fixture
def mock_redis():
    adapter = MagicMock(spec=RedisVectorAdapter)
    adapter.redis_url = "redis://mock:6379"
    adapter.key_prefix = "mock:kb"
    adapter.ping.return_value = True
    adapter.index_chunks.return_value = 0
    adapter.delete_document_chunks.return_value = 0
    return adapter


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.probe_dimension.return_value = 384
    emb.count_tokens.side_effect = lambda t: len(t.split())
    return emb


def test_incomplete_vector_indexing_prevents_false_indexed_status(temp_repo, mock_redis, mock_embeddings):
    """When Redis indexes fewer chunks than generated, document must NOT be marked INDEXED."""
    # Given: index_chunks returns 0 (incomplete write)
    mock_redis.index_chunks.return_value = 0

    service = IngestionService(
        sqlite_repo=temp_repo,
        redis_adapter=mock_redis,
        embeddings_model=mock_embeddings,
        require_vector_store=True,
    )

    sample_md = b"# Test Doc\n\nSection content for consistency testing."
    
    with pytest.raises(IndexingError, match="Vector indexing incomplete"):
        service.ingest_file(filename="test.md", content=sample_md, document_id="doc_fail_1")

    # Verify: document in SQLite must NOT be marked INDEXED
    doc = temp_repo.get_document("doc_fail_1")
    assert doc is not None
    assert doc.status == DocumentStatus.FAILED
    assert "IndexingError" in doc.error_message

    # Verify: compensating rollback on Redis was triggered
    mock_redis.delete_document_chunks.assert_called_with("doc_fail_1")


def test_sqlite_failure_rolls_back_redis_vectors(temp_repo, mock_redis, mock_embeddings):
    """If SQLite commit fails after Redis write, Redis vectors must be rolled back."""
    mock_redis.index_chunks.side_effect = lambda chunks, emb: len(chunks)

    service = IngestionService(
        sqlite_repo=temp_repo,
        redis_adapter=mock_redis,
        embeddings_model=mock_embeddings,
        require_vector_store=True,
    )

    sample_md = b"# Test Doc\n\nValid text that parses into chunks."

    with patch.object(temp_repo, "save_chunks", side_effect=RuntimeError("Disk I/O failure")):
        with pytest.raises(IndexingError, match="SQLite save failed"):
            service.ingest_file(filename="test_io.md", content=sample_md, document_id="doc_io_fail")

    # Verify: compensating delete on Redis was called
    mock_redis.delete_document_chunks.assert_called_with("doc_io_fail")


def test_reconcile_storage_heals_missing_redis_vectors(temp_repo, mock_redis, mock_embeddings):
    """reconcile_storage detects INDEXED documents missing in Redis and reindexes them."""
    # Create an INDEXED document in SQLite with chunks
    doc = Document(
        document_id="doc_reconcile_1",
        filename="rec.md",
        content_type="md",
        sha256="abc123",
        size_bytes=100,
        status=DocumentStatus.INDEXED,
        chunk_count=1,
    )
    temp_repo.save_document(doc)
    chunk = Chunk(
        chunk_id="chk_rec_1",
        document_id="doc_reconcile_1",
        ordinal=0,
        text="Sample chunk content.",
        section="Intro",
        token_count=3,
        sha256="hash1",
    )
    temp_repo.save_chunks([chunk])

    # Configure mock Redis client to report missing chunk key
    client = MagicMock()
    client.scan_iter.return_value = []
    client.exists.return_value = False  # Missing in Redis!
    mock_redis.get_client.return_value = client
    mock_redis.index_chunks.return_value = 1

    service = IngestionService(
        sqlite_repo=temp_repo,
        redis_adapter=mock_redis,
        embeddings_model=mock_embeddings,
    )

    report = service.reconcile_storage()
    assert report["status"] == "REPAIRED"
    assert "doc_reconcile_1" in report["reindexed_documents"]
    mock_redis.index_chunks.assert_called_once()


def test_reconcile_storage_purges_orphaned_redis_keys(temp_repo, mock_redis, mock_embeddings):
    """reconcile_storage purges Redis keys that have no corresponding INDEXED document in SQLite."""
    client = MagicMock()
    client.scan_iter.return_value = [b"mock:kb:orphan_chunk_1"]
    client.type.return_value = b"hash"
    client.hget.return_value = b"deleted_doc_id"
    mock_redis.get_client.return_value = client

    service = IngestionService(
        sqlite_repo=temp_repo,
        redis_adapter=mock_redis,
        embeddings_model=mock_embeddings,
    )

    report = service.reconcile_storage()
    assert report["purged_redis_keys"] == 1
    client.delete.assert_called_with(b"mock:kb:orphan_chunk_1")


def test_reconcile_storage_when_redis_offline(temp_repo, mock_redis, mock_embeddings):
    """reconcile_storage handles unreachable Redis gracefully."""
    mock_redis.ping.return_value = False

    service = IngestionService(
        sqlite_repo=temp_repo,
        redis_adapter=mock_redis,
        embeddings_model=mock_embeddings,
    )

    report = service.reconcile_storage()
    assert report["status"] == "REDIS_UNREACHABLE"
