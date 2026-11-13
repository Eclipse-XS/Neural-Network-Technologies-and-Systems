"""Focused lifecycle regressions using SQLite and an in-memory Redis stand-in."""
import asyncio
from contextlib import contextmanager
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException

from knowledge_copilot.application.ingestion_service import IngestionService
from knowledge_copilot.domain.exceptions import IndexingError
from knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter
from knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository


class ClosingSQLiteRepository(SQLiteRepository):
    @contextmanager
    def _get_connection(self):
        connection = super()._get_connection()
        try:
            with connection:
                yield connection
        finally:
            connection.close()


class IngestionRegressionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = ClosingSQLiteRepository(Path(self.directory.name) / "test.sqlite")
        self.keys = {}
        self.adapter = RedisVectorAdapter()
        client = Mock()
        client.ping.return_value = True
        client.exists.side_effect = lambda key: key in self.keys
        self.adapter._client = client
        self.adapter._store = Mock()

        def add(docs, ids):
            for doc, chunk_id in zip(docs, ids):
                self.keys[f"{self.adapter.key_prefix}:{chunk_id}"] = doc
            return ids

        self.adapter._store.add_documents.side_effect = add
        self.embeddings = Mock(count_tokens=lambda text: len(text.split()))
        self.service = IngestionService(self.repo, self.adapter, self.embeddings, chunk_size=80, chunk_overlap=10)
        self.content = b"# Storage\n\nRedis stores vectors.\n\nSQLite preserves document chunks and metadata.\n\nPersisted text allows safe restoration of missing vectors."

    def test_reindex_restores_chunks_and_preserves_data_on_failure(self):
        from knowledge_copilot.api.routers.documents import reindex_document

        doc = self.service.ingest_file("storage.md", self.content)
        doc.metadata = {"owner": "test"}
        self.repo.save_document(doc)
        chunks = self.repo.get_chunks_for_document(doc.document_id)
        self.keys.clear()
        response = asyncio.run(reindex_document(doc.document_id, self.service))
        self.assertEqual(response.document_id, doc.document_id)
        self.assertEqual(response.chunk_count, len(chunks))
        self.assertEqual(len(self.keys), len(chunks))
        self.assertEqual(self.repo.get_document(doc.document_id).metadata, doc.metadata)
        before = dict(self.keys)
        def fail_after_one(docs, ids):
            self.keys[f"{self.adapter.key_prefix}:{ids[0]}"] = docs[0]
            raise RuntimeError("partial indexing failure")

        self.adapter._store.add_documents.side_effect = fail_after_one
        with self.assertRaises(HTTPException) as failed:
            asyncio.run(reindex_document(doc.document_id, self.service))
        self.assertEqual(failed.exception.status_code, 503)
        self.assertEqual(self.keys, before)
        self.assertEqual(self.repo.get_chunks_for_document(doc.document_id), chunks)
        with self.assertRaises(HTTPException) as missing:
            asyncio.run(reindex_document("missing", self.service))
        self.assertEqual(missing.exception.status_code, 404)

    def test_repeated_ingestion_counts_existing_and_repairs_missing(self):
        doc = self.service.ingest_file("storage.md", self.content)
        chunks = self.repo.get_chunks_for_document(doc.document_id)
        self.adapter._store.add_documents.reset_mock()
        again = self.service.ingest_file("storage.md", self.content)
        self.assertEqual(again.document_id, doc.document_id)
        self.assertEqual(again.chunk_count, len(chunks))
        self.adapter._store.add_documents.assert_not_called()
        self.assertEqual(self.adapter.index_chunks(chunks, self.embeddings), len(chunks))
        self.keys.pop(next(iter(self.keys)))
        restored = self.service.ingest_file("storage.md", self.content)
        self.assertEqual(len(self.keys), restored.chunk_count)
        self.assertEqual(len(self.repo.list_documents()), 1)
        self.assertEqual(self.repo.get_chunks_for_document(doc.document_id), chunks)
        with patch.object(self.adapter, "index_chunks", return_value=0):
            with self.assertRaises(IndexingError):
                self.service.ingest_file("storage.md", self.content)
        self.assertEqual(len(self.keys), len(chunks))
        self.assertEqual(self.repo.get_document(doc.document_id).status, doc.status)


if __name__ == "__main__":
    unittest.main()
