"""Integration tests for document ingestion, idempotency, failure rollback, and deletion."""
import pymupdf
import pytest
from pathlib import Path

from labs.lab07_final_project.src.knowledge_copilot.application.ingestion_service import IngestionService
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import DocumentStatus
from labs.lab07_final_project.src.knowledge_copilot.domain.exceptions import ExtractionError
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository


class MockEmbeddings:
    """Fast deterministic mock embeddings for offline integration testing."""
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.1] * 384

    def probe_dimension(self) -> int:
        return 384


@pytest.fixture
def ingestion_service(tmp_path: Path) -> IngestionService:
    repo = SQLiteRepository(tmp_path / "lifecycle_test.sqlite")
    redis_adapter = RedisVectorAdapter(redis_url="redis://localhost:6380", index_name="lab07_test_idx")
    return IngestionService(
        sqlite_repo=repo,
        redis_adapter=redis_adapter,
        embeddings_model=MockEmbeddings(),
    )


def test_full_document_ingestion_lifecycle(ingestion_service: IngestionService):
    content = b"# Lab 7 Specification\n## Abstract\nLocal RAG knowledge copilot design."
    
    # 1. Ingest document
    doc = ingestion_service.ingest_file(
        filename="spec.md",
        content=content,
        collection="unit_test_collection",
    )
    assert doc.status == DocumentStatus.INDEXED
    assert doc.chunk_count >= 1
    assert doc.filename == "spec.md"

    # 2. Idempotency check: identical upload returns existing indexed doc
    doc_repeat = ingestion_service.ingest_file(
        filename="spec.md",
        content=content,
        collection="unit_test_collection",
    )
    assert doc_repeat.document_id == doc.document_id
    assert doc_repeat.chunk_count == doc.chunk_count

    # 3. List documents
    docs = ingestion_service.list_documents(collection="unit_test_collection")
    assert len(docs) == 1
    assert docs[0].document_id == doc.document_id

    # 4. Delete document
    deleted = ingestion_service.delete_document(doc.document_id)
    assert deleted is True

    # 5. Verify document is gone
    remaining = ingestion_service.list_documents(collection="unit_test_collection")
    assert len(remaining) == 0


def test_scanned_pdf_fails_gracefully_with_explicit_status(ingestion_service: IngestionService):
    """Empty PDF without text layer must fail with FAILED status and not be marked INDEXED."""
    pdf_doc = pymupdf.open()
    pdf_doc.new_page()  # Blank page
    pdf_bytes = pdf_doc.tobytes()
    pdf_doc.close()

    with pytest.raises(ExtractionError, match="Scanned or image-only PDF"):
        ingestion_service.ingest_file(
            filename="scanned_manual.pdf",
            content=pdf_bytes,
            collection="unit_test_collection",
            document_id="scanned_doc_01",
        )

    # Document must be in FAILED status in the database, NEVER left falsely INDEXED
    doc_record = ingestion_service.get_document("scanned_doc_01")
    assert doc_record is not None
    assert doc_record.status == DocumentStatus.FAILED
    assert "OCR required" in (doc_record.error_message or "")
