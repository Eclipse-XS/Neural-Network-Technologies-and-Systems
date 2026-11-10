"""Contract and regression tests for SQLite session synchronization (DEFECT-01)."""
import pytest
from pathlib import Path
from labs.lab07_final_project.src.knowledge_copilot.domain.enums import GroundingStatus
from labs.lab07_final_project.src.knowledge_copilot.domain.models import Turn
from labs.lab07_final_project.src.knowledge_copilot.infrastructure.sqlite_repo import SQLiteRepository


@pytest.fixture
def repo(tmp_path: Path) -> SQLiteRepository:
    return SQLiteRepository(tmp_path / "test_session_memory.sqlite")


def test_authoritative_canonical_answer_matches_persisted_turn(repo: SQLiteRepository):
    """DEFECT-01 requirement: The answer returned to the user MUST match the authoritative assistant turn in DB."""
    session_id = "sess_sync_test"
    turn_id = "turn_001"
    
    # Simulate a raw draft that undergoes post-processing/grounding correction
    raw_draft = "Initial unverified draft with potential hallucination."
    canonical_answer = "Corrected, fully grounded answer with citation [lab04#chunk-123]."
    
    turn = Turn(
        turn_id=turn_id,
        session_id=session_id,
        user_input="What were the results of Lab 4?",
        raw_answer=raw_draft,
        canonical_answer=canonical_answer,
        grounding_status=GroundingStatus.GROUNDED,
        runtime_seconds=1.2,
    )
    
    repo.save_canonical_turn(turn)
    
    # Retrieve persisted session
    session = repo.get_session(session_id)
    assert session is not None
    assert len(session.turns) == 1
    persisted_turn = session.turns[0]
    
    # Authoritative consistency
    assert persisted_turn.canonical_answer == canonical_answer
    assert persisted_turn.canonical_answer != raw_draft
    assert persisted_turn.grounding_status == GroundingStatus.GROUNDED


def test_retries_do_not_produce_duplicate_messages(repo: SQLiteRepository):
    """DEFECT-01 requirement: Retries on the same turn must update in-place without duplicating rows."""
    session_id = "sess_retry_test"
    turn_id = "turn_retry_001"
    
    turn_attempt_1 = Turn(
        turn_id=turn_id,
        session_id=session_id,
        user_input="Calculate the total",
        raw_answer="Attempt 1 output",
        canonical_answer="Attempt 1 answer",
        grounding_status=GroundingStatus.PARTIALLY_SUPPORTED,
    )
    repo.save_canonical_turn(turn_attempt_1)
    
    # Retry on the same turn
    turn_attempt_2 = Turn(
        turn_id=turn_id,
        session_id=session_id,
        user_input="Calculate the total",
        raw_answer="Attempt 2 output",
        canonical_answer="Final retry answer with verified calculation",
        grounding_status=GroundingStatus.GROUNDED,
    )
    repo.save_canonical_turn(turn_attempt_2)
    
    session = repo.get_session(session_id)
    assert len(session.turns) == 1
    assert session.turns[0].canonical_answer == "Final retry answer with verified calculation"
    assert session.turns[0].grounding_status == GroundingStatus.GROUNDED


def test_session_isolation_across_distinct_sessions(repo: SQLiteRepository):
    """Ensure independent sessions cannot leak conversation state."""
    session_a = "sess_alpha"
    session_b = "sess_beta"
    
    repo.save_canonical_turn(
        Turn(
            turn_id="turn_a1",
            session_id=session_a,
            user_input="Alpha confidential query",
            raw_answer="Alpha response",
            canonical_answer="Alpha response",
            grounding_status=GroundingStatus.GROUNDED,
        )
    )
    
    repo.save_canonical_turn(
        Turn(
            turn_id="turn_b1",
            session_id=session_b,
            user_input="Beta query",
            raw_answer="Beta response",
            canonical_answer="Beta response",
            grounding_status=GroundingStatus.GROUNDED,
        )
    )
    
    loaded_a = repo.get_session(session_a)
    loaded_b = repo.get_session(session_b)
    
    assert len(loaded_a.turns) == 1
    assert len(loaded_b.turns) == 1
    assert loaded_a.turns[0].user_input == "Alpha confidential query"
    assert loaded_b.turns[0].user_input == "Beta query"
    assert "Beta" not in loaded_a.turns[0].canonical_answer


def test_failed_requests_do_not_corrupt_subsequent_history(repo: SQLiteRepository):
    """An interrupted or failed turn is preserved with explicit failure status and does not corrupt the session."""
    session_id = "sess_fail_test"
    
    # Successful turn 1
    repo.save_canonical_turn(
        Turn(
            turn_id="turn_ok",
            session_id=session_id,
            user_input="Valid query",
            raw_answer="Valid response",
            canonical_answer="Valid response",
            grounding_status=GroundingStatus.GROUNDED,
        )
    )
    
    # Failed turn 2
    repo.save_canonical_turn(
        Turn(
            turn_id="turn_err",
            session_id=session_id,
            user_input="Malformed or timed out query",
            raw_answer="",
            canonical_answer="Execution timed out; recoverable error.",
            grounding_status=GroundingStatus.EXECUTION_FAILED,
        )
    )
    
    # Successful turn 3
    repo.save_canonical_turn(
        Turn(
            turn_id="turn_resume",
            session_id=session_id,
            user_input="Resumed conversation query",
            raw_answer="Resumed response",
            canonical_answer="Resumed response",
            grounding_status=GroundingStatus.GROUNDED,
        )
    )
    
    session = repo.get_session(session_id)
    assert len(session.turns) == 3
    assert session.turns[0].grounding_status == GroundingStatus.GROUNDED
    assert session.turns[1].grounding_status == GroundingStatus.EXECUTION_FAILED
    assert session.turns[2].grounding_status == GroundingStatus.GROUNDED
    assert session.turns[2].canonical_answer == "Resumed response"
