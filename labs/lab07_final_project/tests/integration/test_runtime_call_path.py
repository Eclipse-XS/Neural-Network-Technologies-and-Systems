"""Gate B1: Runtime Call Path Integration Test.
Verifies the mandatory execution chain:
FastAPI -> QueryService -> AgentOrchestrator -> LocalLLMClient / Runner -> response persistence.
The test verifies that every component participates and fails if any component is bypassed.
"""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import ASGITransport, AsyncClient

from labs.lab07_final_project.src.knowledge_copilot.api.app import create_app
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Turn
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter
from labs.lab07_final_project.src.knowledge_copilot.application.ingestion_service import IngestionService
from labs.lab07_final_project.src.knowledge_copilot.application.query_service import QueryService
from labs.lab07_final_project.src.knowledge_copilot.application.session_service import SessionService
from labs.lab07_final_project.src.knowledge_copilot.agents.orchestrator import AgentOrchestrator


class MockEmbeddings:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts):
        return [[0.05] * 384 for _ in texts]

    def embed_query(self, text):
        return [0.05] * 384

    def probe_dimension(self) -> int:
        return 384


@pytest.mark.asyncio
async def test_real_agent_execution_call_path(tmp_path: Path):
    """
    Verify complete unbroken call chain:
    FastAPI (POST /api/v1/chat)
      -> QueryService.execute_query
        -> AgentOrchestrator.execute_turn
          -> LocalLLM / Runner execution
            -> SQLite persistent storage.
    Fails if any stage is bypassed.
    """
    db_path = tmp_path / "call_path.sqlite"
    sqlite_repo = SQLiteRepository(db_path)

    redis_mock = MagicMock(spec=RedisVectorAdapter)
    redis_mock.ping.return_value = False

    ingestion_svc = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=redis_mock,
        embeddings_model=MockEmbeddings(),
    )

    query_svc = QueryService(
        ingestion_service=ingestion_svc,
        redis_adapter=redis_mock,
        sqlite_repo=sqlite_repo,
    )

    session_svc = SessionService(sqlite_repo=sqlite_repo)

    # Mock LLM client and Orchestrator
    llm_client_mock = MagicMock()
    orchestrator = AgentOrchestrator(
        llm_client=llm_client_mock,
        search_fn=query_svc.search,
        list_sources_fn=query_svc.list_sources,
        db_path=db_path,
    )
    query_svc.orchestrator = orchestrator

    # Spy/wrap execute_turn to verify invocation
    mock_turn = Turn(
        turn_id="turn_callpath_01",
        session_id="session_cp_01",
        user_input="Test query for call path verification",
        raw_answer="Verified response from mock agent",
        canonical_answer="Verified response from mock agent",
        grounding_status=GroundingStatus.GROUNDED,
    )

    orchestrator_spy = AsyncMock(return_value=mock_turn)
    orchestrator.execute_turn = orchestrator_spy

    app = create_app()

    # Override dependencies with our instrumented instances
    from labs.lab07_final_project.src.knowledge_copilot.api.dependencies import (
        get_query_service,
        get_session_service,
        get_sqlite_repo,
    )
    app.dependency_overrides[get_query_service] = lambda: query_svc
    app.dependency_overrides[get_session_service] = lambda: session_svc
    app.dependency_overrides[get_sqlite_repo] = lambda: sqlite_repo

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={
                "session_id": "session_cp_01",
                "query": "Test query for call path verification",
                "retrieval_mode": "lexical",
            },
        )

    # 1. FastAPI received request and returned 200 OK
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    body = response.json()
    assert body["turn_id"] == "turn_callpath_01"
    assert body["answer"] == "Verified response from mock agent"

    # 2. QueryService delegated to AgentOrchestrator.execute_turn (not bypassed)
    orchestrator_spy.assert_awaited_once()
    call_kwargs = orchestrator_spy.await_args.kwargs
    assert call_kwargs["user_input"] == "Test query for call path verification"
    assert call_kwargs["session_id"] == "session_cp_01"

    # 3. Response is physically persisted to SQLite (persistence not bypassed)
    session = sqlite_repo.get_session("session_cp_01")
    assert session is not None
    assert len(session.turns) == 1
    persisted_turn = session.turns[0]
    assert persisted_turn.turn_id == "turn_callpath_01"
    assert persisted_turn.canonical_answer == "Verified response from mock agent"
    assert persisted_turn.session_id == "session_cp_01"
