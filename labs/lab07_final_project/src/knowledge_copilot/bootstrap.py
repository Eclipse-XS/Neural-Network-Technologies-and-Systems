"""Shared, fail-fast corpus initialization for the UI and optional API."""
import hashlib
import json
from pathlib import Path

from .config import LAB_ROOT


def initialize_knowledge():
    from .api.dependencies import get_embeddings, get_ingestion_service, get_redis_adapter

    redis = get_redis_adapter()
    if not redis.ping():
        raise RuntimeError("Redis unavailable; run .\\run.ps1 and check Docker Desktop.")
    try:
        embeddings = get_embeddings()
    except Exception as exc:
        raise RuntimeError(f'Embedding cache unavailable: {exc}. Run .\\setup.ps1 -DownloadEmbedding for the default model; custom models require their configured revision cached locally.') from exc
    if embeddings._offline:
        raise RuntimeError('Synthetic embeddings are not valid for application startup. Cache the configured model with setup.ps1 -DownloadEmbedding.')
    redis.initialize_store(embeddings)
    ingestion = get_ingestion_service()
    directory = LAB_ROOT / "assets/knowledge"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not manifest:
        raise RuntimeError("Knowledge manifest is empty.")
    filenames = set()
    for item in manifest:
        filename = item["workspace_filename"]
        if Path(filename).name != filename:
            raise ValueError("Invalid manifest filename")
        content = (directory / filename).read_bytes()
        if len(content) != item["size_bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise RuntimeError(f"Corpus integrity failure: {filename}")
        filenames.add(filename)
        ingestion.ingest_file(filename, content, "coursework", item["source_id"])
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".txt", ".pdf"} and path.name not in filenames:
            ingestion.ingest_file(path.name, path.read_bytes(), "engineering", path.stem)
    if not ingestion.get_all_indexed_chunks():
        raise RuntimeError("Knowledge corpus has no indexed chunks.")
    return ingestion
