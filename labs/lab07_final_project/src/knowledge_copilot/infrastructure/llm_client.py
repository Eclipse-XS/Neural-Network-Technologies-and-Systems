"""Loopback LM Studio client adapter with preflight verification."""
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import List, Optional, Tuple
from urllib.parse import urlparse
import httpx
import httpx2
from openai import AsyncOpenAI


class LMStudioStatus(str, Enum):
    READY = "READY"
    SERVER_UNREACHABLE = "SERVER_UNREACHABLE"
    NO_MODEL_LOADED = "NO_MODEL_LOADED"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    TOOL_CALLING_UNSUPPORTED = "TOOL_CALLING_UNSUPPORTED"
    INFERENCE_ERROR = "INFERENCE_ERROR"


@dataclass
class LMStudioProbeResult:
    status: LMStudioStatus
    model_id: Optional[str] = None
    available_models: List[str] = field(default_factory=list)
    detail: str = ""
    is_ready: bool = False


def validate_loopback_url(url: str) -> None:
    """Enforce security: strictly loopback-only, credential-free endpoints."""
    p = urlparse(url)
    if p.scheme not in {"http", "https"}:
        raise ValueError(f"Invalid URL scheme '{p.scheme}'; expected http")
    if p.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(f"Endpoint '{url}' must be a loopback address (localhost / 127.0.0.1)")
    if p.username or p.password or p.query or p.fragment:
        raise ValueError("Loopback URL must not include credentials, query, or fragments.")


def probe_lm_studio_readiness(
    base_url: str = "http://localhost:1234/v1",
    required_model: Optional[str] = None,
    timeout: float = 15.0,
) -> LMStudioProbeResult:
    """Perform granular pre-flight probe of local LM Studio instance."""
    try:
        validate_loopback_url(base_url)
    except Exception as exc:
        return LMStudioProbeResult(
            status=LMStudioStatus.SERVER_UNREACHABLE,
            detail=f"Invalid loopback URL: {exc}",
        )

    # 1. Probe /models endpoint
    try:
        models_resp = httpx.get(f"{base_url}/models", timeout=timeout)
        if models_resp.status_code != 200:
            return LMStudioProbeResult(
                status=LMStudioStatus.SERVER_UNREACHABLE,
                detail=f"LM Studio /models returned HTTP {models_resp.status_code}",
            )
        data = models_resp.json().get("data", [])
        available_models = [m["id"] for m in data if isinstance(m, dict) and "id" in m]
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        return LMStudioProbeResult(
            status=LMStudioStatus.SERVER_UNREACHABLE,
            detail=f"Connection refused or timed out connecting to {base_url}: {exc}",
        )
    except Exception as exc:
        return LMStudioProbeResult(
            status=LMStudioStatus.SERVER_UNREACHABLE,
            detail=f"Error connecting to LM Studio at {base_url}: {exc}",
        )

    # 2. Check if any model is loaded
    if not available_models:
        return LMStudioProbeResult(
            status=LMStudioStatus.NO_MODEL_LOADED,
            available_models=[],
            detail="LM Studio server is running, but no models are loaded.",
        )

    # 3. Model discovery / matching
    chosen_model: Optional[str] = None
    if required_model:
        # Exact match
        for m in available_models:
            if required_model.casefold() == m.casefold():
                chosen_model = m
                break
        # Substring match
        if not chosen_model:
            for m in available_models:
                if required_model.casefold() in m.casefold() or m.casefold() in required_model.casefold():
                    chosen_model = m
                    break
        # Alphanumeric normalized match
        if not chosen_model:
            norm_req = re.sub(r"[^a-z0-9]", "", required_model.casefold())
            for m in available_models:
                norm_m = re.sub(r"[^a-z0-9]", "", m.casefold())
                if norm_req in norm_m:
                    chosen_model = m
                    break
        # Substantive stem match (excluding generic model vocabulary)
        if not chosen_model:
            generic = {"model", "chat", "instruct", "base", "text", "v1", "v2", "embed", "embedding", "ai"}
            req_stems = [w for w in re.split(r"[-_/]", required_model.casefold()) if len(w) >= 4 and w not in generic]
            if req_stems:
                for m in available_models:
                    if any(s in m.casefold() for s in req_stems):
                        chosen_model = m
                        break
        if not chosen_model:
            return LMStudioProbeResult(
                status=LMStudioStatus.MODEL_NOT_FOUND,
                available_models=available_models,
                detail=f"Configured model '{required_model}' not found in loaded models: {available_models}",
            )
    else:
        chat_candidates = [m for m in available_models if "embed" not in m.casefold()]
        chosen_model = chat_candidates[0] if chat_candidates else available_models[0]

    # 4. Probe chat completions on the chosen model
    try:
        ping_payload = {
            "model": chosen_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        res = httpx.post(f"{base_url}/chat/completions", json=ping_payload, timeout=timeout)
        if res.status_code != 200:
            err_text = res.text.casefold()
            if "not found" in err_text:
                return LMStudioProbeResult(
                    status=LMStudioStatus.MODEL_NOT_FOUND,
                    model_id=chosen_model,
                    available_models=available_models,
                    detail=f"Model '{chosen_model}' returned 404/not found: {res.text}",
                )
            return LMStudioProbeResult(
                status=LMStudioStatus.INFERENCE_ERROR,
                model_id=chosen_model,
                available_models=available_models,
                detail=f"Inference probe returned HTTP {res.status_code}: {res.text}",
            )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        return LMStudioProbeResult(
            status=LMStudioStatus.SERVER_UNREACHABLE,
            model_id=chosen_model,
            detail=f"Connection timed out during chat completions probe: {exc}",
        )
    except Exception as exc:
        return LMStudioProbeResult(
            status=LMStudioStatus.INFERENCE_ERROR,
            model_id=chosen_model,
            detail=f"Inference probe failed: {exc}",
        )

    # 5. Probe tool calling capability with controlled request and structure validation
    try:
        tool_payload = {
            "model": chosen_model,
            "messages": [
                {"role": "user", "content": "Calculate 2+2 using the add tool."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }],
            "tool_choice": "required",
            "max_tokens": 100,
        }
        tool_timeout = min(15.0, max(5.0, timeout * 2))
        t_res = httpx.post(f"{base_url}/chat/completions", json=tool_payload, timeout=tool_timeout)
        if t_res.status_code != 200:
            return LMStudioProbeResult(
                status=LMStudioStatus.TOOL_CALLING_UNSUPPORTED,
                model_id=chosen_model,
                available_models=available_models,
                detail=f"Model '{chosen_model}' tool probe returned HTTP {t_res.status_code}: {t_res.text}",
            )

        # Strict tool_calls structure validation
        res_data = t_res.json()
        choices = res_data.get("choices", [])
        if not choices:
            return LMStudioProbeResult(
                status=LMStudioStatus.TOOL_CALLING_UNSUPPORTED,
                model_id=chosen_model,
                available_models=available_models,
                detail=f"Model '{chosen_model}' tool probe returned HTTP 200 but empty choices: {res_data}",
            )
        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list):
            return LMStudioProbeResult(
                status=LMStudioStatus.TOOL_CALLING_UNSUPPORTED,
                model_id=chosen_model,
                available_models=available_models,
                detail=f"Model '{chosen_model}' tool probe returned HTTP 200 but missing or invalid tool_calls structure: {msg}",
            )
        first_call = tool_calls[0]
        if not isinstance(first_call, dict) or first_call.get("type") != "function":
            return LMStudioProbeResult(
                status=LMStudioStatus.TOOL_CALLING_UNSUPPORTED,
                model_id=chosen_model,
                available_models=available_models,
                detail=f"Model '{chosen_model}' tool probe returned tool_calls without function type: {first_call}",
            )
        fn_call = first_call.get("function", {})
        if fn_call.get("name") != "add":
            return LMStudioProbeResult(
                status=LMStudioStatus.TOOL_CALLING_UNSUPPORTED,
                model_id=chosen_model,
                available_models=available_models,
                detail=f"Model '{chosen_model}' tool probe invoked unexpected function '{fn_call.get('name')}', expected 'add'",
            )
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        return LMStudioProbeResult(
            status=LMStudioStatus.SERVER_UNREACHABLE,
            model_id=chosen_model,
            available_models=available_models,
            detail=f"Connection error during tool calling probe: {exc}",
        )
    except Exception as exc:
        return LMStudioProbeResult(
            status=LMStudioStatus.TOOL_CALLING_UNSUPPORTED,
            model_id=chosen_model,
            available_models=available_models,
            detail=f"Tool calling probe failed: {exc}",
        )

    return LMStudioProbeResult(
        status=LMStudioStatus.READY,
        model_id=chosen_model,
        available_models=available_models,
        detail=f"Model '{chosen_model}' is online and verified for chat and tools.",
        is_ready=True,
    )


class LocalLLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model_id: str = "mistralai/ministral-3-3b",
        timeout_seconds: float = 120.0,
    ):
        validate_loopback_url(base_url)
        self.base_url = base_url
        self.model_id = model_id
        self.timeout_seconds = timeout_seconds
        self._client: Optional[AsyncOpenAI] = None

    def probe_readiness(self) -> LMStudioProbeResult:
        return probe_lm_studio_readiness(
            base_url=self.base_url,
            required_model=self.model_id,
            timeout=min(5.0, self.timeout_seconds),
        )

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key="lm-studio",
                timeout=self.timeout_seconds,
                max_retries=0,
                http_client=httpx2.AsyncClient(trust_env=False, follow_redirects=False),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    async def is_available(self) -> bool:
        probe = self.probe_readiness()
        return probe.is_ready

    async def list_models(self) -> List[str]:
        client = self.get_client()
        res = await client.models.list()
        return [m.id for m in res.data]

    async def select_model(self) -> str:
        candidates = await self.list_models()
        chat_candidates = [m for m in candidates if "embedding" not in m.lower()]
        if not chat_candidates:
            raise RuntimeError(f"No local chat models found at {self.base_url}: {candidates}")
        
        if self.model_id in chat_candidates:
            return self.model_id
        # Fall back to first available chat model
        self.model_id = chat_candidates[0]
        return self.model_id
