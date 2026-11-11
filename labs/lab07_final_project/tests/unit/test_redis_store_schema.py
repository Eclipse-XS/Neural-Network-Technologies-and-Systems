"""Regression tests for RedisVectorAdapter URL isolation and schema-aware filtering."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from labs.lab07_final_project.src.knowledge_copilot.infrastructure.redis_store import (
    RedisVectorAdapter,
)


def _make_fake_store(captured_filters: list) -> MagicMock:
    store = MagicMock()

    def _search(query: str, k: int = 5, filter=None):
        captured_filters.append(filter)
        return []

    store.similarity_search_with_score.side_effect = _search
    return store


def test_explicit_non_default_redis_url_is_not_overridden(monkeypatch: pytest.MonkeyPatch):
    """Explicit test failure URLs must not be silently hijacked by REDIS_URL fallback."""
    attempted_urls: list[str] = []

    def fake_from_url(url: str, **kwargs):
        attempted_urls.append(url)
        client = MagicMock()
        client.ping.side_effect = ConnectionError("unreachable")
        return client

    monkeypatch.setenv("REDIS_URL", "redis://localhost:6380")
    monkeypatch.setattr("redis.Redis.from_url", fake_from_url)
    adapter = RedisVectorAdapter(redis_url="redis://127.0.0.1:65432")
    with pytest.raises(ConnectionError, match="unreachable"):
        adapter.get_client()
    assert attempted_urls == ["redis://127.0.0.1:65432"]


@pytest.mark.parametrize(
    ("ft_attributes", "expect_doc_id", "expect_quality"),
    [
        (
            [
                [b"identifier", b"source_id", b"attribute", b"source_id", b"type", b"TAG"],
                [b"identifier", b"evidence_quality", b"attribute", b"evidence_quality", b"type", b"TAG"],
            ],
            False,
            True,
        ),
        (
            [
                [b"identifier", b"document_id", b"attribute", b"document_id", b"type", b"TAG"],
                [b"identifier", b"source_id", b"attribute", b"source_id", b"type", b"TAG"],
                [b"identifier", b"evidence_quality", b"attribute", b"evidence_quality", b"type", b"TAG"],
            ],
            True,
            True,
        ),
        (
            [
                [b"identifier", b"source_id", b"attribute", b"source_id", b"type", b"TAG"],
            ],
            False,
            False,
        ),
    ],
)
def test_dense_search_filters_only_indexed_tag_fields(
    ft_attributes: list,
    expect_doc_id: bool,
    expect_quality: bool,
):
    """dense_search omits unindexed TAG fields (e.g. legacy indexes lacking document_id)."""
    adapter = RedisVectorAdapter(redis_url="redis://127.0.0.1:65432")
    fake_client = MagicMock()
    fake_client.ping.return_value = True
    fake_client.execute_command.return_value = [
        b"index_name",
        b"lab07_course_kb_idx",
        b"attributes",
        ft_attributes,
    ]
    adapter._client = fake_client

    captured_filters: list = []
    adapter._store = _make_fake_store(captured_filters)

    adapter.dense_search("test query", k=3, document_id="doc_123", filter_quoted_outputs=True)

    assert len(captured_filters) == 1
    rendered = str(captured_filters[0])
    assert "@source_id:" in rendered
    assert ("@document_id:" in rendered) is expect_doc_id
    assert ("@evidence_quality:" in rendered) is expect_quality
