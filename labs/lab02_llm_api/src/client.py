from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import Any

from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "mistralai/ministral-3-3b"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("Base URL must not be empty.")
    return normalized


def create_client(
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> OpenAI:
    if isinstance(timeout, bool) or not isinstance(timeout, Real) or timeout <= 0:
        raise ValueError("Timeout must be a positive number of seconds.")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("Max retries must be a non-negative integer.")
    return OpenAI(
        base_url=normalize_base_url(base_url),
        api_key="lm-studio",
        timeout=float(timeout),
        max_retries=max_retries,
    )


def validate_generation_settings(
    model: str,
    temperature: float,
    max_tokens: int,
) -> None:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("Model id must not be empty.")
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise ValueError("Temperature must be a number between 0 and 2.")
    if not 0 <= temperature <= 2:
        raise ValueError("Temperature must be between 0 and 2.")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise ValueError("Max tokens must be a positive integer.")
    if max_tokens <= 0:
        raise ValueError("Max tokens must be a positive integer.")


def list_model_ids(client: Any) -> list[str]:
    models = client.models.list()
    data = getattr(models, "data", None)
    if not isinstance(data, Sequence):
        raise RuntimeError("LM Studio returned a malformed models response.")
    return [model.id for model in data if getattr(model, "id", None)]


def ensure_model_available(client: Any, model: str) -> list[str]:
    model_ids = list_model_ids(client)
    if model not in model_ids:
        available = ", ".join(model_ids) if model_ids else "none"
        raise ValueError(
            f"Model '{model}' is not available in LM Studio. "
            f"Available model ids: {available}."
        )
    return model_ids


def request_chat_completion(
    client: Any,
    *,
    model: str,
    messages: Sequence[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    validate_generation_settings(model, temperature, max_tokens)
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("LM Studio returned no completion choices.")
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError("LM Studio returned a choice without a message payload.")
    content = getattr(message, "content", None)
    if content is None or not str(content).strip():
        raise RuntimeError("LM Studio returned an empty assistant response.")
    return str(content).strip()
