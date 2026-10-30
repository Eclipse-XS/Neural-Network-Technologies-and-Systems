"""Streamlit AppTest functional E2E tests over live services."""
import socket
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

WEB_APP_PATH = str((Path(__file__).resolve().parent.parent / "web_app.py"))


def is_live_available() -> bool:
    try:
        with socket.create_connection(("localhost", 6380), timeout=1):
            return True
    except OSError:
        return False


live_required = pytest.mark.skipif(
    not is_live_available(),
    reason="Requires live Redis on localhost:6380"
)


@live_required
def test_streamlit_initial_render_and_catalog():
    at = AppTest.from_file(WEB_APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Coursework Knowledge Assistant"
    subheaders = [s.value for s in at.sidebar.subheader]
    assert "Session" in subheaders
    assert "Indexed documents" in subheaders
    assert bool(at.chat_input)


@live_required
def test_streamlit_live_e2e_question_and_answer():
    at = AppTest.from_file(WEB_APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    
    # Submit real question
    at.chat_input[0].set_value("What model was used in Lab 6?").run(timeout=120)
    assert not at.exception
    assert len(at.chat_message) >= 2
    
    user_msg = at.chat_message[0]
    assert user_msg.name == "user"
    assert "What model was used in Lab 6?" in user_msg.markdown[0].value
    
    assistant_msg = at.chat_message[1]
    assert assistant_msg.name == "assistant"
    answer_text = assistant_msg.markdown[0].value
    assert "mistralai/ministral-3-3b" in answer_text
    assert assistant_msg.json
    raw_details = assistant_msg.json[0].value
    import json
    details = json.loads(raw_details) if isinstance(raw_details, str) else raw_details
    assert "tools_used" in details
    assert "search_knowledge" in details["tools_used"]
    assert "lab06" in details.get("sources_retrieved", [])


@live_required
def test_streamlit_session_controls():
    at = AppTest.from_file(WEB_APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception
    
    initial_session_id = at.session_state["session_id"]
    assert initial_session_id.startswith("web_")
    
    # Click new session
    at.sidebar.button[0].click().run(timeout=30)
    assert not at.exception
    new_session_id = at.session_state["session_id"]
    assert new_session_id != initial_session_id
    assert len(at.session_state["messages"]) == 0
