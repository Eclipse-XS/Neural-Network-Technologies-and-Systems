"""Only HTTP calls to an already configured local server; no LLM weight loading."""
import json
import os
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def list_models(settings, opener=urlopen):
    headers = {"Authorization": "Bearer " + os.getenv("LM_STUDIO_API_KEY", "lm-studio")}
    request = Request(settings.base_url.rstrip("/") + "/models", headers=headers)
    try:
        with opener(request, timeout=5) as response:
            models = [entry["id"] for entry in json.load(response)["data"]]
    except (URLError, HTTPError, TimeoutError, OSError, ValueError, KeyError) as exc:
        raise RuntimeError("LM Studio unavailable: load a local chat model, start Developer → "
                           "Local Server on port 1234, then retry /v1/models. "
                           f"Original error: {exc}") from exc
    return models


def create_llm(settings):
    models = list_models(settings)
    if not settings.model or settings.model not in models:
        raise ValueError(f"Set LM_STUDIO_MODEL to a chat-model ID from /v1/models: {models}")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(base_url=settings.base_url, model=settings.model,
                      api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
                      temperature=0, max_tokens=800, timeout=120, max_retries=0)
