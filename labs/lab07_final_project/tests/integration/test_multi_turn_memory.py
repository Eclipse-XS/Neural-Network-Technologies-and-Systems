"""Comprehensive multi-turn memory regression, session restart, isolation, and concurrency tests (Gate B2)."""
import asyncio
from pathlib import Path
import pytest
from agents import SQLiteSession

from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus, RetrievalMode
from labs.lab07_final_project.src.knowledge_copilot.domain.models import EvidenceChunk, Score, ScoreType, Turn
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository
from labs.lab07_final_project.src.knowledge_copilot.application.session_service import SessionService
from labs.lab07_final_project.src.knowledge_copilot.application.query_service import QueryService
from labs.lab07_final_project.src.knowledge_copilot.agents.orchestrator import AgentOrchestrator


class DeterministicMockLLM:
    """Mock LLM client that provides deterministic responses echoing prior context."""
    def __init__(self, model_id: str = "mock-chat-model"):
        self.model_id = model_id

    def get_client(self):
        class MockClient:
            pass
        return MockClient()


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
# 1. 25 Deterministic Multi-Turn Regression Cases (Gate B2)
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


@pytest.mark.parametrize("session_id, q1, a1, q2, a2", MULTI_TURN_REGRESSION_CASES)
def test_25_deterministic_multi_turn_cases(
    test_db_path: Path,
    sqlite_repo: SQLiteRepository,
    session_id: str,
    q1: str,
    a1: str,
    q2: str,
    a2: str,
):
    """Verify that model-visible history is preserved across turns and persisted in SQLite."""
    # 1. Turn 1 execution
    sdk_session = SQLiteSession(session_id, db_path=test_db_path)
    asyncio.run(sdk_session.add_items([
        {"role": "user", "content": q1},
        {"role": "assistant", "content": a1},
    ]))
    items_t1 = asyncio.run(sdk_session.get_items())
    assert len(items_t1) == 2
    assert items_t1[0]["content"] == q1
    assert items_t1[1]["content"] == a1
    sdk_session.close()

    # Turn 1 persistence in SQLiteRepository
    t1 = Turn(
        turn_id=f"t1_{session_id}",
        session_id=session_id,
        user_input=q1,
        raw_answer=a1,
        canonical_answer=a1,
        grounding_status=GroundingStatus.GROUNDED,
    )
    sqlite_repo.save_canonical_turn(t1)

    # 2. Turn 2 execution (model sees Turn 1 history)
    sdk_session2 = SQLiteSession(session_id, db_path=test_db_path)
    prior_items = asyncio.run(sdk_session2.get_items())
    assert len(prior_items) == 2, f"Turn 2 must see Turn 1 history for {session_id}"
    assert prior_items[0]["content"] == q1

    asyncio.run(sdk_session2.add_items([
        {"role": "user", "content": q2},
        {"role": "assistant", "content": a2},
    ]))
    items_t2 = asyncio.run(sdk_session2.get_items())
    assert len(items_t2) == 4
    sdk_session2.close()

    # Turn 2 persistence in SQLiteRepository
    t2 = Turn(
        turn_id=f"t2_{session_id}",
        session_id=session_id,
        user_input=q2,
        raw_answer=a2,
        canonical_answer=a2,
        grounding_status=GroundingStatus.GROUNDED,
    )
    sqlite_repo.save_canonical_turn(t2)

    # Verify canonical session state
    session_model = sqlite_repo.get_session(session_id)
    assert session_model is not None
    assert len(session_model.turns) == 2
    assert session_model.turns[0].user_input == q1
    assert session_model.turns[1].user_input == q2


# ============================================================================
# 2. Session Restart & Persistence Recovery
# ============================================================================

def test_session_restart_restores_complete_history(test_db_path: Path, sqlite_repo: SQLiteRepository):
    """Restarting application / closing database connection must preserve all turns."""
    session_id = "restart_test_sess"
    
    # Session lifecycle phase 1: Active turns
    s1 = SQLiteSession(session_id, db_path=test_db_path)
    asyncio.run(s1.add_items([
        {"role": "user", "content": "Tell me about ADR 001."},
        {"role": "assistant", "content": "ADR 001 defines the dual vector and SQLite storage."},
    ]))
    s1.close()

    t1 = Turn(
        turn_id="turn_r1",
        session_id=session_id,
        user_input="Tell me about ADR 001.",
        raw_answer="ADR 001 defines the dual vector and SQLite storage.",
        canonical_answer="ADR 001 defines the dual vector and SQLite storage.",
        grounding_status=GroundingStatus.GROUNDED,
    )
    sqlite_repo.save_canonical_turn(t1)

    # Simulation of service restart: create fresh instances pointing to existing sqlite
    fresh_repo = SQLiteRepository(test_db_path)
    fresh_sdk_session = SQLiteSession(session_id, db_path=test_db_path)
    
    restored_items = asyncio.run(fresh_sdk_session.get_items())
    assert len(restored_items) == 2
    assert "ADR 001" in restored_items[0]["content"]

    restored_session = fresh_repo.get_session(session_id)
    assert restored_session is not None
    assert len(restored_session.turns) == 1
    assert restored_session.turns[0].canonical_answer == t1.canonical_answer
    fresh_sdk_session.close()


# ============================================================================
# 3. Session Isolation (Zero Cross-Talk)
# ============================================================================

def test_session_isolation_guarantees_no_leakage(test_db_path: Path, sqlite_repo: SQLiteRepository):
    """Sessions A and B must be strictly isolated: items in A must never leak to B."""
    sA = SQLiteSession("session_A", db_path=test_db_path)
    sB = SQLiteSession("session_B", db_path=test_db_path)

    asyncio.run(sA.add_items([
        {"role": "user", "content": "Secret token in session A: X999_ALPHA"},
        {"role": "assistant", "content": "Stored confidential session A token."},
    ]))

    items_b = asyncio.run(sB.get_items())
    assert len(items_b) == 0, "Session B must have empty history initially"

    asyncio.run(sB.add_items([
        {"role": "user", "content": "Public query in session B"},
        {"role": "assistant", "content": "Session B response"},
    ]))

    items_a = asyncio.run(sA.get_items())
    items_b = asyncio.run(sB.get_items())

    assert len(items_a) == 2
    assert len(items_b) == 2
    assert "X999_ALPHA" in items_a[0]["content"]
    assert "X999_ALPHA" not in str(items_b)

    sA.close()
    sB.close()


# ============================================================================
# 4. Concurrent Request Execution
# ============================================================================

def test_concurrent_sessions_under_wal_mode(test_db_path: Path, sqlite_repo: SQLiteRepository):
    """Multiple concurrent sessions writing simultaneously must not encounter database locks."""
    async def run_concurrent():
        async def worker(worker_id: int):
            sess_id = f"worker_sess_{worker_id}"
            sdk_sess = SQLiteSession(sess_id, db_path=test_db_path)
            await sdk_sess.add_items([
                {"role": "user", "content": f"Worker {worker_id} query"},
                {"role": "assistant", "content": f"Worker {worker_id} answer"},
            ])
            sdk_sess.close()

            turn = Turn(
                turn_id=f"turn_w_{worker_id}",
                session_id=sess_id,
                user_input=f"Worker {worker_id} query",
                raw_answer=f"Worker {worker_id} answer",
                canonical_answer=f"Worker {worker_id} answer",
                grounding_status=GroundingStatus.GROUNDED,
            )
            sqlite_repo.save_canonical_turn(turn)
            return True

        # Run 10 workers concurrently
        tasks = [worker(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        return results

    results = asyncio.run(run_concurrent())
    assert all(results)
    assert len(results) == 10

    # Verify all 10 sessions were persisted without lock errors
    for i in range(10):
        s = sqlite_repo.get_session(f"worker_sess_{i}")
        assert s is not None
        assert len(s.turns) == 1
