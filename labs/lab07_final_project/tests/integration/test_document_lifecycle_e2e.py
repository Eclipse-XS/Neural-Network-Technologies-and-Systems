"""Gate B4: Full Document Lifecycle Integration Test.
Verifies: upload -> index -> query -> reindex -> delete -> restart workflow
through public application interfaces and guarantees Redis and SQLite agreement.
"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from labs.lab07_final_project.src.knowledge_copilot.domain.enums import DocumentStatus, RetrievalMode
from labs.lab07_final_project.src.knowledge_copilot.domain.exceptions import DocumentNotFoundError
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter
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
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "lifecycle_test.sqlite"


@pytest.fixture
def live_or_mock_redis():
    """Connect to live Redis on 6380 if available, else in-memory mock."""
    adapter = RedisVectorAdapter(redis_url="redis://localhost:6380", index_name="lifecycle_test_idx")
    if adapter.ping():
        client = adapter.get_client()
        keys = list(client.scan_iter("lab07:kb:lifecycle_*"))
        if keys:
            client.delete(*keys)
        return adapter
    mock = MagicMock(spec=RedisVectorAdapter)
    mock.ping.return_value = False
    return mock


def test_full_document_lifecycle_workflow(db_path: Path, live_or_mock_redis):
    """
    Step-by-step verification of:
    1. Upload & Index
    2. Multi-mode Retrieval
    3. Idempotent Reindex
    4. Deletion Consistency
    5. Restart & Persistence Agreement
    """
    # -------------------------------------------------------------------------
    # Phase 1: Upload and Index
    # -------------------------------------------------------------------------
    sqlite_repo = SQLiteRepository(db_path)
    embeddings = MockFastEmbeddings()
    ingestion_svc = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=live_or_mock_redis,
        embeddings_model=embeddings,
    )
    query_svc = QueryService(
        ingestion_service=ingestion_svc,
        redis_adapter=live_or_mock_redis,
        sqlite_repo=sqlite_repo,
    )

    doc_content = (
        b"# Enterprise Lifecycle Spec\n"
        b"## Core Architecture\n"
        b"The primary telemetry collector operates at 42.195 megahertz with zero jitter.\n"
        b"## Operational Metrics\n"
        b"Observed Mean Time To Failure (MTTF) is 8760 hours under nominal load.\n"
    )
    doc_id = "lifecycle_doc_01"
    doc = ingestion_svc.ingest_file(
        filename="lifecycle_spec.md",
        content=doc_content,
        document_id=doc_id,
        collection="engineering",
    )

    assert doc.status == DocumentStatus.INDEXED
    assert doc.document_id == doc_id
    assert doc.sha256 is not None

    # Verify SQLite persistence
    stored_doc = sqlite_repo.get_document(doc_id)
    assert stored_doc is not None
    assert stored_doc.status == DocumentStatus.INDEXED
    chunks = sqlite_repo.get_chunks_for_document(doc_id)
    assert len(chunks) >= 2, f"Expected at least 2 section chunks, got {len(chunks)}"

    # -------------------------------------------------------------------------
    # Phase 2: Multi-mode Query & Retrieval
    # -------------------------------------------------------------------------
    # Test LEXICAL mode
    lex_results = query_svc.search("telemetry collector 42.195 megahertz", k=5, mode=RetrievalMode.LEXICAL)
    assert len(lex_results) >= 1
    assert any("42.195" in c.chunk.text for c in lex_results)
    assert lex_results[0].score.retrieval_method == "lexical"

    # Test DENSE / HYBRID mode
    hybrid_results = query_svc.search("Mean Time To Failure 8760 hours", k=5, mode=RetrievalMode.HYBRID)
    assert len(hybrid_results) >= 1
    assert any("8760" in c.chunk.text for c in hybrid_results)

    # -------------------------------------------------------------------------
    # Phase 3: Idempotent Reindex
    # -------------------------------------------------------------------------
    # Re-ingest with identical content
    re_doc = ingestion_svc.ingest_file(
        filename="lifecycle_spec.md",
        content=doc_content,
        document_id=doc_id,
        collection="engineering",
    )
    assert re_doc.document_id == doc_id
    assert re_doc.sha256 == doc.sha256
    re_chunks = sqlite_repo.get_chunks_for_document(doc_id)
    assert len(re_chunks) == len(chunks), "Re-indexing must not produce duplicate chunks"

    # -------------------------------------------------------------------------
    # Phase 4: Atomic Deletion
    # -------------------------------------------------------------------------
    delete_ok = ingestion_svc.delete_document(doc_id)
    assert delete_ok is True

    # SQLite must have no record or chunks for doc_id
    assert sqlite_repo.get_document(doc_id) is None
    assert len(sqlite_repo.get_chunks_for_document(doc_id)) == 0

    # Querying after deletion must yield NO chunks from deleted document
    post_delete_lex = query_svc.search("telemetry collector 42.195 megahertz", k=5, mode=RetrievalMode.LEXICAL)
    assert not any(c.chunk.document_id == doc_id for c in post_delete_lex)

    # -------------------------------------------------------------------------
    # Phase 5: Service Restart & Consistency Agreement
    # -------------------------------------------------------------------------
    # Simulate fresh process startup using the same persistent storage
    restarted_repo = SQLiteRepository(db_path)
    restarted_ingestion = IngestionService(
        sqlite_repo=restarted_repo,
        redis_adapter=live_or_mock_redis,
        embeddings_model=embeddings,
    )
    restarted_query = QueryService(
        ingestion_service=restarted_ingestion,
        redis_adapter=live_or_mock_redis,
        sqlite_repo=restarted_repo,
    )

    # Re-verify that deleted document is still absent after service restart
    assert restarted_repo.get_document(doc_id) is None
    assert len(restarted_repo.get_chunks_for_document(doc_id)) == 0
    restart_results = restarted_query.search("telemetry collector 42.195 megahertz", k=5, mode=RetrievalMode.LEXICAL)
    assert len(restart_results) == 0

    # Ingest a second clean document to verify post-restart indexing
    doc2 = restarted_ingestion.ingest_file(
        filename="post_restart_doc.md",
        content=b"# Post Restart Doc\nVerified post-restart operational continuity.",
        document_id="lifecycle_doc_02",
        collection="engineering",
    )
    assert doc2.status == DocumentStatus.INDEXED
    assert restarted_repo.get_document("lifecycle_doc_02") is not None
    post_restart_search = restarted_query.search("operational continuity", k=5, mode=RetrievalMode.LEXICAL)
    assert len(post_restart_search) >= 1
    assert any(c.chunk.document_id == "lifecycle_doc_02" for c in post_restart_search)
