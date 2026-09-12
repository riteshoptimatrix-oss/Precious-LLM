"""
Precious Edu LLM — Health Check Endpoint Tests
"""

import pytest


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    response = await async_client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Precious AI"
    assert data["database"] == "connected"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_database_health_endpoint(async_client):
    response = await async_client.get("/api/health/db")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "connected"
    assert data["database_name"] in ("precious_ai", "precious_ai_test", "precious_edu_llm")
