"""Gate P0-3: Retrieval Mode Propagation Test.
Verifies that POST /api/v1/chat respects retrieval_mode without global mutation:
- 'dense' invokes redis_adapter.dense_search, not bm25.search
- 'lexical' invokes bm25.search, not redis_adapter.dense_search
- 'hybrid' invokes both backends
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
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import DocumentStatus, RetrievalMode
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Chunk, Document
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository


def make_tool_call_completions(query_text: str):
    """Creates a 2-step model response: 1) tool call to search_knowledge, 2) final answer."""
    tc = ChatCompletionMessageToolCall(
        id="call_search_01",
        type="function",
        function=Function(name="search_knowledge", arguments=json.dumps({"query": query_text})),
    )
    msg1 = ChatCompletionMessage(role="assistant", content=None, tool_calls=[tc])
    resp1 = ChatCompletion(
        id="cmpl_tool",
        choices=[Choice(finish_reason="tool_calls", index=0, message=msg1)],
        created=int(time.time()),
        model="mistralai/ministral-3-3b",
        object="chat.completion",
    )

    msg2 = ChatCompletionMessage(role="assistant", content="The search concluded successfully.")
    resp2 = ChatCompletion(
        id="cmpl_final",
        choices=[Choice(finish_reason="stop", index=0, message=msg2)],
        created=int(time.time()),
        model="mistralai/ministral-3-3b",
        object="chat.completion",
    )
    return [resp1, resp2]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode_name,expected_dense,expected_lexical", [
    ("dense", True, False),
    ("lexical", False, True),
    ("hybrid", True, True),
])
async def test_retrieval_mode_propagation_to_tools(
    tmp_path: Path, mode_name: str, expected_dense: bool, expected_lexical: bool
):
    db_path = tmp_path / f"modes_{mode_name}.sqlite"
    sqlite_repo = SQLiteRepository(db_path)

    # Seed one indexed chunk so query service has something to search
    doc = Document(
        document_id="doc_test_01",
        filename="test.md",
        content_type="text/markdown",
        sha256="hash01",
        size_bytes=100,
        chunk_count=1,
        status=DocumentStatus.INDEXED,
    )
    sqlite_repo.save_document(doc)
    chunk = Chunk(
        chunk_id="chunk_01",
        document_id="doc_test_01",
        ordinal=0,
        text="The latency target for the system is 250 milliseconds.",
        section="Performance",
        token_count=10,
        sha256="chunkhash01",
    )
    sqlite_repo.save_chunks([chunk])

    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    redis_mock.dense_search.return_value = [(chunk, 0.15)]

    embeddings_mock = MagicMock()
    embeddings_mock.count_tokens.return_value = 10
    embeddings_mock.embed_documents.return_value = [[0.1] * 384]
    embeddings_mock.embed_query.return_value = [0.1] * 384

    ingestion_svc = IngestionService(
        sqlite_repo=sqlite_repo,
        redis_adapter=redis_mock,
        embeddings_model=embeddings_mock,
        require_vector_store=False,
    )

    query_svc = QueryService(
        ingestion_service=ingestion_svc,
        redis_adapter=redis_mock,
        sqlite_repo=sqlite_repo,
        default_retrieval_mode=RetrievalMode.HYBRID,
    )

    # Spy BM25 search
    real_bm25_search = query_svc.bm25.search
    bm25_mock = MagicMock(side_effect=real_bm25_search)
    query_svc.bm25.search = bm25_mock

    # Controlled LLM Client returning tool call -> final answer
    llm_client_mock = MagicMock()
    llm_client_mock.model_id = "mistralai/ministral-3-3b"
    async_openai_mock = MagicMock()
    async_openai_mock.base_url = "http://localhost:1234/v1"
    async_openai_mock.chat.completions.create = AsyncMock(
        side_effect=make_tool_call_completions("system latency")
    )
    llm_client_mock.get_client.return_value = async_openai_mock

    orchestrator = AgentOrchestrator(
        llm_client=llm_client_mock,
        search_fn=query_svc.search,
        list_sources_fn=query_svc.list_sources,
        db_path=db_path,
    )
    query_svc.orchestrator = orchestrator

    session_svc = SessionService(sqlite_repo=sqlite_repo)

    app = create_app()
    app.dependency_overrides[get_query_service] = lambda: query_svc
    app.dependency_overrides[get_session_service] = lambda: session_svc
    app.dependency_overrides[get_sqlite_repo] = lambda: sqlite_repo

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        resp = await client.post(
            "/api/v1/chat",
            json={
                "session_id": f"sess_{mode_name}",
                "query": "What is the system latency target?",
                "retrieval_mode": mode_name,
            },
        )

    assert resp.status_code == 200, f"Error {resp.status_code}: {resp.text}"

    # Assert backend invocation matches selected retrieval_mode
    if expected_dense:
        assert redis_mock.dense_search.call_count >= 1, (
            f"Expected dense_search to be called for mode={mode_name}, but was not called"
        )
    else:
        assert redis_mock.dense_search.call_count == 0, (
            f"Expected dense_search NOT to be called for mode={mode_name}, but called {redis_mock.dense_search.call_count} times"
        )

    if expected_lexical:
        assert bm25_mock.call_count >= 1, (
            f"Expected bm25.search to be called for mode={mode_name}, but was not called"
        )
    else:
        assert bm25_mock.call_count == 0, (
            f"Expected bm25.search NOT to be called for mode={mode_name}, but called {bm25_mock.call_count} times"
        )

    # Ensure default retrieval mode was NOT permanently mutated
    assert query_svc.default_retrieval_mode == RetrievalMode.HYBRID
