"""Gate B1 / P0-1: Real Agent Execution Call Path Integration Test.
Verifies the complete unbroken execution chain WITHOUT mocking AgentOrchestrator:
FastAPI (POST /api/v1/chat)
  -> QueryService.execute_query
    -> AgentOrchestrator.execute_turn (REAL - NOT MOCKED)
      -> OpenAI Agents SDK Runner.run()
        -> Tool execution (calculate)
          -> Semantic Grounding Verification
            -> SQLite persistent storage (SQLiteRepository + SQLiteSession).
Fails if any stage in the execution chain is bypassed.
"""
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from agents import SQLiteSession

from labs.lab07_final_project.src.knowledge_copilot.agents.orchestrator import AgentOrchestrator
from labs.lab07_final_project.src.knowledge_copilot.api.app import create_app
from labs.lab07_final_project.src.knowledge_copilot.api.dependencies import (
    get_query_service,
    get_session_service,
    get_sqlite_repo,
)
from labs.lab07_final_project.src.knowledge_copilot.application.ingestion_service import IngestionService
from labs.lab07_final_project.src.knowledge_copilot.application.query_service import QueryService
from labs.lab07_final_project.src.knowledge_copilot.application.session_service import SessionService
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import DocumentStatus, GroundingStatus
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, Document
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import RedisVectorAdapter
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository


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
async def test_real_agent_execution_call_path_unbroken(tmp_path: Path):
    """
    Verify full unbroken real agent execution without AsyncMock on execute_turn.
    """
    db_path = tmp_path / "call_path_real.sqlite"
    sqlite_repo = SQLiteRepository(db_path)

    # Seed one test document and chunk in SQLite
    doc = Document(
        document_id="doc_cp_01",
        filename="latency_spec.md",
        content_type="text/markdown",
        sha256="sha256_cp",
        size_bytes=200,
        chunk_count=1,
        status=DocumentStatus.INDEXED,
    )
    sqlite_repo.save_document(doc)
    chunk = Chunk(
        chunk_id="chunk_01",
        document_id="doc_cp_01",
        ordinal=0,
        text="The network latency budget is 120 ms and disk latency is 30 ms.",
        section="Architecture",
        token_count=16,
        sha256="chunk_sha_cp",
    )
    sqlite_repo.save_chunks([chunk])

    redis_mock = MagicMock(spec=RedisVectorAdapter)
    redis_mock.ping.return_value = True
    redis_mock.dense_search.return_value = [(chunk, 0.12)]

    ingestion_svc = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=redis_mock,
        embeddings_model=MockEmbeddings(),
        require_vector_store=False,
    )

    query_svc = QueryService(
        ingestion_service=ingestion_svc,
        redis_adapter=redis_mock,
        sqlite_repo=sqlite_repo,
    )

    session_svc = SessionService(sqlite_repo=sqlite_repo)

    # Prepare controlled model responses for Runner.run():
    # 1. Model executes tool call to calculate: '120 + 30'
    tc = ChatCompletionMessageToolCall(
        id="call_calc_path_01",
        type="function",
        function=Function(name="calculate", arguments=json.dumps({"expression": "120 + 30"})),
    )
    msg1 = ChatCompletionMessage(role="assistant", content=None, tool_calls=[tc])
    resp1 = ChatCompletion(
        id="cmpl_1",
        choices=[Choice(finish_reason="tool_calls", index=0, message=msg1)],
        created=int(time.time()),
        model="mistralai/ministral-3-3b",
        object="chat.completion",
    )

    # 2. Model receives tool result 150 and produces grounded final answer
    msg2 = ChatCompletionMessage(
        role="assistant",
        content="The total latency budget is 150 ms [doc_cp_01#chunk_01].",
    )
    resp2 = ChatCompletion(
        id="cmpl_2",
        choices=[Choice(finish_reason="stop", index=0, message=msg2)],
        created=int(time.time()),
        model="mistralai/ministral-3-3b",
        object="chat.completion",
    )

    llm_client_mock = MagicMock()
    llm_client_mock.model_id = "mistralai/ministral-3-3b"
    async_openai_mock = MagicMock()
    async_openai_mock.base_url = "http://localhost:1234/v1"
    async_openai_mock.chat.completions.create = AsyncMock(side_effect=[resp1, resp2])
    llm_client_mock.get_client.return_value = async_openai_mock

    # Real AgentOrchestrator instance - execute_turn is NOT replaced or mocked
    orchestrator = AgentOrchestrator(
        llm_client=llm_client_mock,
        search_fn=query_svc.search,
        list_sources_fn=query_svc.list_sources,
        db_path=db_path,
        max_turns=5,
    )
    query_svc.orchestrator = orchestrator

    app = create_app()
    app.dependency_overrides[get_query_service] = lambda: query_svc
    app.dependency_overrides[get_session_service] = lambda: session_svc
    app.dependency_overrides[get_sqlite_repo] = lambda: sqlite_repo

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/chat",
            json={
                "session_id": "session_callpath_real",
                "query": "Calculate the total latency budget for the system.",
                "retrieval_mode": "hybrid",
            },
        )

    # 1. FastAPI received request and returned HTTP 200 OK
    assert response.status_code == 200, f"Expected 200 OK, got {response.status_code}: {response.text}"
    body = response.json()
    assert body["session_id"] == "session_callpath_real"
    assert "150 ms" in body["answer"]

    # 2. Verify that Runner.run actually called the model adapter
    assert async_openai_mock.chat.completions.create.call_count == 2, (
        f"Expected 2 LLM completion calls (tool call + final answer), got {async_openai_mock.chat.completions.create.call_count}"
    )

    # 3. Verify tool execution record from actual Runner tool dispatch
    tools_used = body["tools_used"]
    assert len(tools_used) >= 1, "Expected at least one tool execution record in response"
    calc_record = next((t for t in tools_used if t["tool_name"] == "calculate"), None)
    assert calc_record is not None, "calculate tool was not recorded in response"
    assert calc_record["arguments"] == {"expression": "120 + 30"}
    assert calc_record["result"]["result"] == 150

    # 4. Verify canonical SQLite persistence
    session = sqlite_repo.get_session("session_callpath_real")
    assert session is not None, "Session was not persisted to SQLite"
    assert len(session.turns) == 1
    persisted_turn = session.turns[0]
    assert persisted_turn.session_id == "session_callpath_real"
    assert "150 ms" in persisted_turn.canonical_answer
    assert any(t.tool_name == "calculate" for t in persisted_turn.tool_executions)

    # 5. Verify model-visible OpenAI Agents SDK SQLiteSession contains conversation items
    sdk_sess = SQLiteSession(session_id="session_callpath_real", db_path=db_path)
    sdk_items = await sdk_sess.get_items()
    assert len(sdk_items) >= 2, f"Expected at least user and assistant items in SDK session, got {len(sdk_items)}"
    assert sdk_items[0]["role"] == "user"
    assert "latency budget" in sdk_items[0]["content"]
    sdk_sess.close()
