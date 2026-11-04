"""Prometheus telemetry metrics for Knowledge Copilot."""
from prometheus_client import Counter, Gauge, Histogram, REGISTRY

# Request metrics
RAG_REQUESTS_TOTAL = Counter(
    "rag_requests_total",
    "Total RAG and agent queries processed",
    ["status", "grounding_status"],
)

RAG_REQUEST_DURATION_SECONDS = Histogram(
    "rag_request_duration_seconds",
    "End-to-end duration of RAG queries",
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 15.0, 30.0, 60.0],
)

RAG_RETRIEVAL_DURATION_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    "Duration of vector and hybrid retrieval",
    ["mode"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0],
)

RAG_RETRIEVAL_EMPTY_TOTAL = Counter(
    "rag_retrieval_empty_total",
    "Total queries returning zero evidence chunks",
)

RAG_TOOL_CALLS_TOTAL = Counter(
    "rag_tool_calls_total",
    "Total tool calls executed",
    ["tool_name", "origin"],
)

RAG_CITATION_VALIDATION_FAILURES_TOTAL = Counter(
    "rag_citation_validation_failures_total",
    "Total citation validation failures",
    ["failure_type"],
)

RAG_INDEXED_DOCUMENTS = Gauge(
    "rag_indexed_documents",
    "Current count of active indexed documents",
)

RAG_INDEXED_CHUNKS = Gauge(
    "rag_indexed_chunks",
    "Current count of active vector chunks",
)

RAG_INGESTION_FAILURES_TOTAL = Counter(
    "rag_ingestion_failures_total",
    "Total document ingestion failures",
    ["error_type"],
)
