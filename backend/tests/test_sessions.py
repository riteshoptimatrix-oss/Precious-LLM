"""
Precious Edu LLM — Session API Tests
"""

import pytest


@pytest.mark.asyncio
async def test_create_session(async_client):
    response = await async_client.post("/api/sessions", json={"title": "Test Session"})
    assert response.status_code == 201
    data = response.json()
    assert "session_id" in data
    assert data["title"] == "Test Session"


@pytest.mark.asyncio
async def test_get_session(async_client):
    create_res = await async_client.post("/api/sessions", json={"title": "Retrievable Session"})
    session_id = create_res.json()["session_id"]

    response = await async_client.get(f"/api/sessions/{session_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["title"] == "Retrievable Session"
    assert "messages" in data


@pytest.mark.asyncio
async def test_list_sessions(async_client):
    await async_client.post("/api/sessions", json={"title": "Session 1"})
    await async_client.post("/api/sessions", json={"title": "Session 2"})

    response = await async_client.get("/api/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data
    assert data["total"] >= 2


@pytest.mark.asyncio
async def test_delete_session(async_client):
    create_res = await async_client.post("/api/sessions", json={"title": "To Delete"})
    session_id = create_res.json()["session_id"]

    delete_res = await async_client.delete(f"/api/sessions/{session_id}")
    assert delete_res.status_code == 200

    get_res = await async_client.get(f"/api/sessions/{session_id}")
    assert get_res.status_code == 404


@pytest.mark.asyncio
async def test_invalid_session_get(async_client):
    response = await async_client.get("/api/sessions/non-existent-uuid")
    assert response.status_code == 404
