"""Prometheus telemetry metrics for Knowledge Copilot."""
from prometheus_client import Counter, Gauge, Histogram, REGISTRY


def _get_or_create(metric_cls, name, documentation, *args, **kwargs):
    """Idempotently retrieve or register a Prometheus collector across module alias imports."""
    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing
    return metric_cls(name, documentation, *args, **kwargs)


# Request metrics
RAG_REQUESTS_TOTAL = _get_or_create(
    Counter,
    "rag_requests_total",
    "Total RAG and agent queries processed",
    ["status", "grounding_status"],
)

RAG_REQUEST_DURATION_SECONDS = _get_or_create(
    Histogram,
    "rag_request_duration_seconds",
    "End-to-end duration of RAG queries",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0],
)

RAG_RETRIEVAL_DURATION_SECONDS = _get_or_create(
    Histogram,
    "rag_retrieval_duration_seconds",
    "Duration of vector and hybrid retrieval",
    ["mode"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
)

RAG_RETRIEVAL_EMPTY_TOTAL = _get_or_create(
    Counter,
    "rag_retrieval_empty_total",
    "Total queries returning zero evidence chunks",
)

RAG_TOOL_CALLS_TOTAL = _get_or_create(
    Counter,
    "rag_tool_calls_total",
    "Total tool calls executed",
    ["tool_name", "origin"],
)

RAG_CITATION_VALIDATION_FAILURES_TOTAL = _get_or_create(
    Counter,
    "rag_citation_validation_failures_total",
    "Total citation validation failures",
    ["failure_type"],
)

RAG_INDEXED_DOCUMENTS = _get_or_create(
    Gauge,
    "rag_indexed_documents",
    "Current count of active indexed documents",
)

RAG_INDEXED_CHUNKS = _get_or_create(
    Gauge,
    "rag_indexed_chunks",
    "Current count of active vector chunks",
)

RAG_INGESTION_FAILURES_TOTAL = _get_or_create(
    Counter,
    "rag_ingestion_failures_total",
    "Total document ingestion failures",
    ["error_type"],
)
