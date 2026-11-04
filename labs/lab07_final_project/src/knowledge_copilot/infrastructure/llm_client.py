"""Loopback LM Studio client adapter with preflight verification."""
from urllib.parse import urlparse
from typing import List, Optional, Tuple
import httpx2
from openai import AsyncOpenAI


def validate_loopback_url(url: str) -> None:
    """Enforce security: strictly loopback-only, credential-free endpoints."""
    p = urlparse(url)
    if p.scheme not in {"http", "https"}:
        raise ValueError(f"Invalid URL scheme '{p.scheme}'; expected http")
    if p.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(f"Endpoint '{url}' must be a loopback address (localhost / 127.0.0.1)")
    if p.username or p.password or p.query or p.fragment:
        raise ValueError("Loopback URL must not include credentials, query, or fragments.")


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
        try:
            client = self.get_client()
            models = await client.models.list()
            return bool(models.data)
        except Exception:
            return False

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
