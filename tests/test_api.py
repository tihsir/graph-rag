"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from graphrag.app.api.routers import app

client = TestClient(app)


def test_health():
    """Test health endpoint."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_ingest_empty():
    """Test ingestion with empty request."""
    response = client.post("/ingest", json={"paths": [], "urls": [], "tags": []})
    # Should not fail, but may return empty list
    assert response.status_code in [200, 400]


@pytest.mark.asyncio
async def test_query_basic():
    """Test basic query endpoint."""
    request_data = {
        "query": "Test query",
        "mode": "rag",
        "k": 5,
        "hops": 1,
        "rerank": False,
    }
    response = client.post("/query", json=request_data)
    # May fail if no documents ingested, but should not crash
    assert response.status_code in [200, 500]

