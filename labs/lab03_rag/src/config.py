"""Explicit, lab-local configuration; importing never contacts a service."""
from dataclasses import dataclass, field
from pathlib import Path
import os
from urllib.parse import urlparse

LAB_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = LAB_ROOT / "data/raw/csv"
    output_dir: Path = LAB_ROOT / "outputs"
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    base_url: str = field(default_factory=lambda: os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1"))
    model: str = field(default_factory=lambda: os.getenv("LM_STUDIO_MODEL", ""))
    top_k: int = 5

    def __post_init__(self):
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("LM_STUDIO_BASE_URL must point to a local OpenAI-compatible server")
        if parsed.path.rstrip("/") != "/v1" or parsed.query or parsed.fragment:
            raise ValueError("LM_STUDIO_BASE_URL must end in /v1")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
