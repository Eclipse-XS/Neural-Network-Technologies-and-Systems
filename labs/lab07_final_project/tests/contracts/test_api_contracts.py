"""FastAPI contract and schema validation tests."""
import io
import pytest
from starlette.testclient import TestClient

from labs.lab07_final_project.src.knowledge_copilot.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_live_endpoint(client: TestClient):
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "live"
    assert "redis_connected" in data
    assert "sqlite_connected" in data
    assert "llm_available" in data
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-MS" in response.headers


def test_health_ready_endpoint(client: TestClient):
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in {"ready", "degraded"}


def test_metrics_prometheus_endpoint(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_requests_total" in response.text
    assert "rag_indexed_documents" in response.text


def test_list_documents_contract(client: TestClient):
    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data
    assert isinstance(data["documents"], list)


def test_document_upload_rejects_empty_file(client: TestClient):
    empty_file = io.BytesIO(b"")
    response = client.post(
        "/api/v1/documents",
        files={"file": ("empty.md", empty_file, "text/markdown")},
    )
    assert response.status_code == 400
    assert "is empty" in response.json()["detail"]


def test_document_upload_rejects_unsupported_extension(client: TestClient):
    fake_exe = io.BytesIO(b"MZ12345")
    response = client.post(
        "/api/v1/documents",
        files={"file": ("payload.exe", fake_exe, "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_chat_endpoint_validates_input(client: TestClient):
    # Query must not be empty
    response = client.post("/api/v1/chat", json={"query": ""})
    assert response.status_code == 422
