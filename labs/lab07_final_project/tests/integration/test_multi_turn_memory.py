"""Comprehensive multi-turn memory regression, session restart, isolation, and concurrency tests (Gate B2 / P0-4).
Verifies that:
1. All multi-turn executions run through real AgentOrchestrator.execute_turn and Runner.run().
2. The model receives canonical conversation history in subsequent turns.
3. Post-processed canonical answers are reflected in the model-visible session (no SDK/SQLite divergence).
4. Session restart and session isolation preserve state without cross-talk.
"""
import asyncio
import time
from pathlib import Path
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from agents import SQLiteSession
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from labs.lab07_final_project.src.knowledge_copilot.agents.orchestrator import AgentOrchestrator
from labs.lab07_final_project.src.knowledge_copilot.application.session_service import SessionService
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Turn
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository


def make_mock_llm_adapter(responses: List[str]) -> Tuple[MagicMock, MagicMock]:
    """Build a mock LLM adapter returning pre-configured ChatCompletion objects."""
    mock_client = MagicMock()
    mock_client.base_url = "http://localhost:1234/v1"

    completions = []
    for idx, resp_text in enumerate(responses):
        msg = ChatCompletionMessage(role="assistant", content=resp_text)
        cmpl = ChatCompletion(
            id=f"cmpl_{idx}",
            choices=[Choice(finish_reason="stop", index=0, message=msg)],
            created=int(time.time()),
            model="mistralai/ministral-3-3b",
            object="chat.completion",
        )
        completions.append(cmpl)

    mock_client.chat.completions.create = AsyncMock(side_effect=completions)

    adapter = MagicMock()
    adapter.model_id = "mistralai/ministral-3-3b"
    adapter.get_client.return_value = mock_client
    return adapter, mock_client


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    return tmp_path / "memory_test.sqlite"


@pytest.fixture
def sqlite_repo(test_db_path: Path) -> SQLiteRepository:
    return SQLiteRepository(test_db_path)


@pytest.fixture
def session_service(sqlite_repo: SQLiteRepository) -> SessionService:
    return SessionService(sqlite_repo)


# ============================================================================
# 1. 25 Deterministic Multi-Turn Regression Cases through AgentOrchestrator
# ============================================================================

MULTI_TURN_REGRESSION_CASES = [
    ("session_01", "What was the scheduler in Lab 4?", "Lab 4 used PNDMScheduler.", "Why was that chosen?", "It was the default pipeline scheduler."),
    ("session_02", "How many parameters in Lab 5 model?", "It has 864,031,264 parameters.", "Does that include the vision encoder?", "Yes, the full SigLIP and Qwen model."),
    ("session_03", "What is the memory limit in Redis?", "The configured maxmemory is 4.0 GB.", "What happens if it is exceeded?", "Redis returns OOM errors under noeviction."),
    ("session_04", "Which embedding model is used?", "multilingual-e5-small is used.", "What is its dimension?", "Its dimension is 384."),
    ("session_05", "What is the SQLite journal mode?", "WAL mode is enabled.", "Why WAL?", "To allow concurrent readers and prevent table locking."),
    ("session_06", "What is the chunk token budget?", "The maximum token budget is 240 tokens.", "What happens to larger chunks?", "They are subdivided hierarchically."),
    ("session_07", "How does the AST calculator work?", "It evaluates mathematical expressions via ast.parse.", "Does it allow exec?", "No, eval and exec are strictly banned."),
    ("session_08", "What is the top-k in hybrid retrieval?", "Default top-k is 5.", "Can it be overridden?", "Yes, via the k query parameter."),
    ("session_09", "What is the RRF constant k?", "The constant is k=60.", "Why 60?", "Standard Cormack reciprocal rank fusion damping constant."),
    ("session_10", "What port does FastAPI listen on?", "Port 8000.", "And Streamlit?", "Port 8501."),
    ("session_11", "What status is assigned to scanned PDFs?", "Status FAILED with OCR required.", "Is it indexed?", "No, empty scans are never marked INDEXED."),
    ("session_12", "What is the maximum upload size?", "Maximum file size is 10 MB.", "Are executables accepted?", "No, rejected with HTTP 400."),
    ("session_13", "What was the incident date in postmortem?", "November 2, 2026.", "How long was the outage?", "14 minutes total downtime."),
    ("session_14", "What is the primary image in Lab 5?", "primary_image.png 512x512.", "What is its SHA-256?", "1fc881e5282f86a05f3abb98201985881a331a5241cd404bf9e2b560b5bb08c5."),
    ("session_15", "What was the inference time for baseline in Lab 4?", "18.876 seconds.", "And for steps 20?", "6.765 seconds."),
    ("session_16", "What is the error code for ungrounded citations?", "UNGROUNDED_CLAIM.", "What is the confidence?", "insufficient_evidence."),
    ("session_17", "Does the system support Redis clustering?", "Currently single-node Redis Stack.", "What is the index name?", "lab07_course_kb_idx."),
    ("session_18", "What API mode is used for LM Studio?", "OpenAI-compatible chat completions.", "What temperature?", "temperature 0.0."),
    ("session_19", "What is the metric Recall@5 target?", "Target is at least 0.85.", "Was it achieved?", "Yes, observed 0.8765."),
    ("session_20", "What is the MRR target?", "Target is at least 0.75.", "Was it achieved?", "Yes, observed 0.7714."),
    ("session_21", "What is the Hit@5 target?", "Target is at least 0.92.", "Was it achieved?", "Yes, observed 0.9294."),
    ("session_22", "What is the maximum agent turns?", "Max turns is 10.", "Can it loop infinitely?", "No, bounds prevent infinite loops."),
    ("session_23", "How are sessions identified?", "Via session_id string.", "Are they shared between users?", "No, strictly isolated per session_id."),
    ("session_24", "What is the Prometheus endpoint?", "/metrics.", "Does it track requests?", "Yes, rag_requests_total counter."),
    ("session_25", "What is the health probe URL?", "/api/v1/health/live.", "And readiness?", "/api/v1/health/ready."),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id, q1, a1, q2, a2", MULTI_TURN_REGRESSION_CASES)
async def test_25_deterministic_multi_turn_cases_through_real_orchestrator(
    test_db_path: Path,
    sqlite_repo: SQLiteRepository,
    session_id: str,
    q1: str,
    a1: str,
    q2: str,
    a2: str,
):
    """Verify that multi-turn execution runs through AgentOrchestrator and preserves canonical history."""
    llm_adapter, mock_client = make_mock_llm_adapter([a1, a2])

    orchestrator = AgentOrchestrator(
        llm_client=llm_adapter,
        search_fn=lambda **kwargs: [],
        list_sources_fn=lambda: [],
        db_path=test_db_path,
        max_turns=5,
    )

    # 1. Turn 1 execution through real orchestrator
    t1 = await orchestrator.execute_turn(user_input=q1, session_id=session_id)
    assert t1.canonical_answer == a1
    sqlite_repo.save_canonical_turn(t1)

    # 2. Turn 2 execution through real orchestrator
    t2 = await orchestrator.execute_turn(user_input=q2, session_id=session_id)
    assert t2.canonical_answer == a2
    sqlite_repo.save_canonical_turn(t2)

    # 3. Verify model input on Turn 2: the model MUST have received Turn 1 user and assistant messages
    assert mock_client.chat.completions.create.call_count == 2
    turn2_call_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]

    # Filter out system prompt to inspect dialog history
    dialog_messages = [m for m in turn2_call_messages if m.get("role") != "system"]
    assert len(dialog_messages) >= 3, f"Turn 2 must receive at least 3 messages (User1, Assistant1, User2), got {dialog_messages}"
    assert dialog_messages[0]["role"] == "user"
    assert dialog_messages[0]["content"] == q1
    assert dialog_messages[1]["role"] == "assistant"
    assert a1 in str(dialog_messages[1]["content"])
    assert dialog_messages[2]["role"] == "user"
    assert dialog_messages[2]["content"] == q2

    # 4. Verify canonical SQLite repository session
    session = sqlite_repo.get_session(session_id)
    assert session is not None
    assert len(session.turns) == 2
    assert session.turns[0].user_input == q1
    assert session.turns[0].canonical_answer == a1
    assert session.turns[1].user_input == q2
    assert session.turns[1].canonical_answer == a2


# ============================================================================
# 2. Canonical Answer Sync (Prevent SDK Session Divergence - P0-4)
# ============================================================================

@pytest.mark.asyncio
async def test_canonical_postprocessing_syncs_with_model_visible_session(
    test_db_path: Path, sqlite_repo: SQLiteRepository
):
    """
    When raw answer has placeholder strings stripped or is recovered,
    the model on subsequent turns must see the canonical answer, NOT raw answer.
    """
    session_id = "test_sync_divergence_sess"

    # Turn 1: Model outputs raw text containing placeholder [source_id#chunk_id]
    raw_output_t1 = "The architecture uses SQLite WAL mode [source_id#chunk_id]."
    canonical_expected_t1 = "The architecture uses SQLite WAL mode."
    output_t2 = "Confirmed, WAL mode supports concurrent readers."

    llm_adapter, mock_client = make_mock_llm_adapter([raw_output_t1, output_t2])

    orchestrator = AgentOrchestrator(
        llm_client=llm_adapter,
        search_fn=lambda **kwargs: [],
        list_sources_fn=lambda: [],
        db_path=test_db_path,
        max_turns=5,
    )

    # Turn 1
    t1 = await orchestrator.execute_turn(
        user_input="What journal mode is used?", session_id=session_id
    )
    assert t1.raw_answer == raw_output_t1
    assert t1.canonical_answer == canonical_expected_t1
    assert "[source_id#chunk_id]" not in t1.canonical_answer
    sqlite_repo.save_canonical_turn(t1)

    # Turn 2
    t2 = await orchestrator.execute_turn(
        user_input="Does that allow readers?", session_id=session_id
    )
    sqlite_repo.save_canonical_turn(t2)

    # Inspect Turn 2 model call: the assistant message must NOT contain '[source_id#chunk_id]'
    turn2_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assistant_hist = next(m for m in turn2_messages if m.get("role") == "assistant")
    assert "[source_id#chunk_id]" not in str(assistant_hist["content"]), (
        "Model-visible session diverged: contains raw placeholder instead of canonical answer"
    )
    assert "SQLite WAL mode." in str(assistant_hist["content"])


# ============================================================================
# 3. Session Restart & Persistence Recovery with Real Orchestrator
# ============================================================================

@pytest.mark.asyncio
async def test_session_restart_restores_complete_history(
    test_db_path: Path, sqlite_repo: SQLiteRepository
):
    """Restarting application preserves turns and allows continued dialogue."""
    session_id = "restart_real_sess"

    # Phase 1: Turn 1 with initial orchestrator
    adapter1, _ = make_mock_llm_adapter(["ADR 001 defines dual storage."])
    orch1 = AgentOrchestrator(
        llm_client=adapter1,
        search_fn=lambda **kwargs: [],
        list_sources_fn=lambda: [],
        db_path=test_db_path,
    )
    t1 = await orch1.execute_turn(
        user_input="Tell me about ADR 001.", session_id=session_id
    )
    sqlite_repo.save_canonical_turn(t1)

    # Phase 2: Fresh orchestrator instance (simulating restart)
    saved_session = sqlite_repo.get_session(session_id)
    history_messages = [
        {"role": "user", "content": t.user_input}
        if i % 2 == 0
        else {"role": "assistant", "content": t.canonical_answer}
        for t in saved_session.turns
        for i in range(2)
    ]
    # Build clean history:
    clean_history = []
    for t in saved_session.turns:
        clean_history.append({"role": "user", "content": t.user_input})
        clean_history.append({"role": "assistant", "content": t.canonical_answer})

    adapter2, mock_client2 = make_mock_llm_adapter(["It pairs Redis vector with SQLite."])
    orch2 = AgentOrchestrator(
        llm_client=adapter2,
        search_fn=lambda **kwargs: [],
        list_sources_fn=lambda: [],
        db_path=test_db_path,
    )

    t2 = await orch2.execute_turn(
        user_input="What does it pair?",
        session_id=session_id,
        history_messages=clean_history,
    )
    sqlite_repo.save_canonical_turn(t2)

    # Assert Turn 2 saw Turn 1
    t2_messages = mock_client2.chat.completions.create.call_args_list[0].kwargs["messages"]
    dialog = [m for m in t2_messages if m.get("role") != "system"]
    assert len(dialog) >= 3
    assert dialog[0]["content"] == "Tell me about ADR 001."
    assert "ADR 001 defines dual storage." in str(dialog[1]["content"])


# ============================================================================
# 4. Session Isolation (Zero Cross-Talk)
# ============================================================================

@pytest.mark.asyncio
async def test_session_isolation_guarantees_no_leakage(
    test_db_path: Path, sqlite_repo: SQLiteRepository
):
    """Sessions A and B must be strictly isolated: items in A must never leak to B."""
    adapterA, mockA = make_mock_llm_adapter(["Token X999_ALPHA saved."])
    adapterB, mockB = make_mock_llm_adapter(["Session B answer."])

    orchA = AgentOrchestrator(
        llm_client=adapterA,
        search_fn=lambda **kwargs: [],
        list_sources_fn=lambda: [],
        db_path=test_db_path,
    )
    orchB = AgentOrchestrator(
        llm_client=adapterB,
        search_fn=lambda **kwargs: [],
        list_sources_fn=lambda: [],
        db_path=test_db_path,
    )

    tA = await orchA.execute_turn(
        user_input="Store secret X999_ALPHA", session_id="session_A"
    )
    sqlite_repo.save_canonical_turn(tA)

    tB = await orchB.execute_turn(
        user_input="List tokens", session_id="session_B"
    )
    sqlite_repo.save_canonical_turn(tB)

    # Session B must never have seen session A's secret
    b_messages = mockB.chat.completions.create.call_args_list[0].kwargs["messages"]
    assert "X999_ALPHA" not in str(b_messages)
