"""
Precious Edu LLM — Chat API & Message Persistence Tests
"""

import pytest


@pytest.mark.asyncio
async def test_chat_pipeline(async_client):
    create_res = await async_client.post("/api/sessions", json={"title": "Chat Test Session"})
    session_id = create_res.json()["session_id"]

    chat_payload = {
        "session_id": session_id,
        "message": "hello world"
    }

    response = await async_client.post("/api/chat", json=chat_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert isinstance(data["response"], str) and len(data["response"]) > 0
    assert data["role"] == "assistant"

    # Verify message persistence
    session_res = await async_client.get(f"/api/sessions/{session_id}")
    assert session_res.status_code == 200
    messages = session_res.json()["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hello world"
    assert messages[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_chat_empty_message_validation(async_client):
    create_res = await async_client.post("/api/sessions", json={"title": "Validation Test"})
    session_id = create_res.json()["session_id"]

    response = await async_client.post("/api/chat", json={"session_id": session_id, "message": "   "})
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_chat_invalid_session(async_client):
    response = await async_client.post("/api/chat", json={"session_id": "non-existent-id", "message": "hello"})
    assert response.status_code == 404
