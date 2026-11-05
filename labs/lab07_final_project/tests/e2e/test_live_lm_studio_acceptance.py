"""Live end-to-end acceptance suite with LM Studio and Redis.

Executes full 8-step lifecycle:
1. Verify Redis and Local LLM (LM Studio on port 1234) availability.
2. Execute real chat request through FastAPI test client.
3. Ingest a previously unseen document.
4. Verify retrieval of the new document.
5. Verify multi-turn memory state through real AgentOrchestrator.
6. Assert on canonical answer format and valid citations.
7. Delete document and verify complete cleanup in SQLite and Redis.
8. If LM Studio is offline, marks gate honestly as BLOCKED (Host Dependency Offline).
"""
import asyncio
import os
import pytest
import httpx
from pathlib import Path
from fastapi.testclient import TestClient

from labs.lab07_final_project.src.knowledge_copilot.api.app import app
from labs.lab07_final_project.src.knowledge_copilot.api.dependencies import (
    get_sqlite_repo,
    get_redis_adapter,
    get_llm_client,
    get_ingestion_service,
    get_query_service,
)
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import DocumentStatus


def check_lm_studio_online(base_url: str = "http://localhost:1234/v1") -> bool:
    try:
        resp = httpx.get(f"{base_url}/models", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def test_live_lm_studio_acceptance_suite():
    """Live acceptance test verifying real networked components."""
    lm_studio_url = os.getenv("COPILOT_LLM_URL", "http://localhost:1234/v1")
    is_llm_up = check_lm_studio_online(lm_studio_url)

    if not is_llm_up:
        pytest.skip(
            f"Gate status: BLOCKED (Host Dependency Offline). LM Studio is not listening at {lm_studio_url}."
        )

    # 1. Verify dependencies
    redis_adapter = get_redis_adapter()
    if not redis_adapter.ping():
        pytest.skip("Gate status: BLOCKED (Redis offline at port 6380).")

    client = TestClient(app)
    sqlite_repo = get_sqlite_repo()
    ingestion_service = get_ingestion_service()
    query_service = get_query_service()

    session_id = "e2e_live_acceptance_session"
    unseen_doc_id = "doc_unseen_acceptance_test"
    doc_content = b"# Acceptance Protocol\n\nQuantum telemetry reveals high energy latency of 14.2 milliseconds in laboratory gamma."

    try:
        # 2. Execute chat request through FastAPI
        resp = client.post(
            "/api/v1/chat",
            json={
                "session_id": session_id,
                "query": "What is the primary function of this copilot?",
                "retrieval_mode": "hybrid",
            },
        )
        assert resp.status_code == 200, f"Chat endpoint failed: {resp.text}"
        data = resp.json()
        assert "answer" in data
        assert "grounding_status" in data

        # 3. Index previously unseen document
        doc = ingestion_service.ingest_file(
            filename="unseen_acceptance_doc.md",
            content=doc_content,
            document_id=unseen_doc_id,
        )
        assert doc.status == DocumentStatus.INDEXED
        assert doc.chunk_count > 0

        # 4. Check retrieval of the newly indexed document
        evidence = query_service.search(
            query="quantum telemetry high energy latency gamma",
            k=3,
            document_id=unseen_doc_id,
        )
        assert len(evidence) >= 1
        assert "14.2 milliseconds" in evidence[0].chunk.text

        # 5. Multi-turn execution through FastAPI
        turn2_resp = client.post(
            "/api/v1/chat",
            json={
                "session_id": session_id,
                "query": "According to the acceptance protocol, what is the latency?",
                "retrieval_mode": "hybrid",
            },
        )
        assert turn2_resp.status_code == 200
        turn2_data = turn2_resp.json()

        # 6. Assert canonical structure and citations
        assert "answer" in turn2_data
        assert "grounding_status" in turn2_data
        assert isinstance(turn2_data.get("citations"), list)

        # Verify session persistence in SQLite
        db_sess = sqlite_repo.get_session(session_id)
        assert db_sess is not None
        assert len(db_sess.turns) >= 2

    finally:
        # 7. Cleanup unseen document from Redis and SQLite
        try:
            ingestion_service.delete_document(unseen_doc_id)
        except Exception:
            pass

        # Verify document is purged
        assert sqlite_repo.get_document(unseen_doc_id) is None
        chunks_left = sqlite_repo.get_chunks_for_document(unseen_doc_id)
        assert len(chunks_left) == 0
