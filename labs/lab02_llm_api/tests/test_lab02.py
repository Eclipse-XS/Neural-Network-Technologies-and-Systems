from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from labs.lab02_llm_api.src.cli import write_transcript
from labs.lab02_llm_api.src.client import (
    ensure_model_available,
    request_chat_completion,
    validate_generation_settings,
)


class FakeCompletions:
    def __init__(self, content="Local response"):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def fake_client(model_ids=("mistralai/ministral-3-3b",), content="Local response"):
    completions = FakeCompletions(content)
    client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: SimpleNamespace(
                data=[SimpleNamespace(id=model_id) for model_id in model_ids]
            )
        ),
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_must_be_in_supported_range(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        validate_generation_settings("model", temperature, 100)


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, True])
def test_max_tokens_must_be_a_positive_integer(max_tokens):
    with pytest.raises(ValueError, match="Max tokens"):
        validate_generation_settings("model", 0.7, max_tokens)


def test_model_id_must_not_be_empty():
    with pytest.raises(ValueError, match="Model id"):
        validate_generation_settings("   ", 0.7, 100)


def test_valid_generation_settings_are_accepted():
    validate_generation_settings("mistralai/ministral-3-3b", 0.7, 256)


def test_exact_model_is_available():
    client, _ = fake_client()
    assert ensure_model_available(client, "mistralai/ministral-3-3b") == [
        "mistralai/ministral-3-3b"
    ]


def test_missing_model_reports_available_ids():
    client, _ = fake_client(("loaded/model",))
    with pytest.raises(ValueError, match="loaded/model"):
        ensure_model_available(client, "missing/model")


def test_chat_request_contains_full_history_and_generation_settings():
    client, completions = fake_client()
    history = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Follow-up"},
    ]

    answer = request_chat_completion(
        client,
        model="mistralai/ministral-3-3b",
        messages=history,
        temperature=0.2,
        max_tokens=200,
    )

    assert answer == "Local response"
    assert completions.calls == [
        {
            "model": "mistralai/ministral-3-3b",
            "messages": history,
            "temperature": 0.2,
            "max_tokens": 200,
        }
    ]


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_assistant_content_is_rejected(content):
    client, _ = fake_client(content=content)
    with pytest.raises(RuntimeError, match="empty assistant response"):
        request_chat_completion(
            client,
            model="mistralai/ministral-3-3b",
            messages=[{"role": "user", "content": "Question"}],
            temperature=0.7,
            max_tokens=100,
        )


def test_transcript_contains_configuration_and_actual_turns(tmp_path):
    history = [
        {"role": "system", "content": "Відповідай стисло."},
        {"role": "user", "content": "Що таке LLM?"},
        {"role": "assistant", "content": "Це велика мовна модель."},
    ]
    path = write_transcript(
        history,
        base_url="http://localhost:1234/v1",
        model="mistralai/ministral-3-3b",
        system_prompt="Відповідай стисло.",
        temperature=0.2,
        max_tokens=200,
        output_dir=tmp_path,
        timestamp=datetime(2026, 10, 1, 19, 30, tzinfo=timezone.utc),
    )

    assert path is not None
    content = path.read_text(encoding="utf-8")
    assert "mistralai/ministral-3-3b" in content
    assert "Temperature: 0.2" in content
    assert "## User\n\nЩо таке LLM?" in content
    assert "## Assistant\n\nЦе велика мовна модель." in content
