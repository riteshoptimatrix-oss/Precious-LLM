"""
Integration test for Phase 9: Full End-to-End Chat Flow with Custom Phase 8 LLM
"""

import pytest


@pytest.mark.asyncio
async def test_llm_health_endpoint(async_client):
    res = await async_client.get("/api/health/llm")
    assert res.status_code == 200
    data = res.json()
    assert data["is_ready"] is True
    assert data["status"] == "ready"
    assert "model_version" in data
    assert "parameters" in data


@pytest.mark.asyncio
async def test_end_to_end_multi_turn_chat(async_client):
    # 1. Create Session
    create_res = await async_client.post("/api/sessions", json={"title": "End to End LLM Session"})
    assert create_res.status_code == 201
    session_id = create_res.json()["session_id"]

    # 2. User Turn 1: Greeting
    res1 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "Hello"})
    assert res1.status_code == 200
    resp1 = res1.json()
    assert resp1["session_id"] == session_id
    assert len(resp1["response"]) > 0

    # 3. User Turn 2: Name Statement
    res2 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "My name is Ritesh."})
    assert res2.status_code == 200

    # 4. User Turn 3: Question
    res3 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "What is a student visa?"})
    assert res3.status_code == 200

    # 5. Verify MongoDB session history persistence
    history_res = await async_client.get(f"/api/sessions/{session_id}")
    assert history_res.status_code == 200
    messages = history_res.json()["messages"]
    assert len(messages) == 6  # 3 user messages + 3 assistant responses
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "My name is Ritesh."


@pytest.mark.asyncio
async def test_session_isolation(async_client):
    # Session A
    res_a = await async_client.post("/api/sessions", json={"title": "Session A"})
    sess_a = res_a.json()["session_id"]
    await async_client.post("/api/chat", json={"session_id": sess_a, "message": "My name is Ritesh."})

    # Session B
    res_b = await async_client.post("/api/sessions", json={"title": "Session B"})
    sess_b = res_b.json()["session_id"]
    await async_client.post("/api/chat", json={"session_id": sess_b, "message": "What is my name?"})

    # History of B must NOT contain Session A messages
    hist_b = await async_client.get(f"/api/sessions/{sess_b}")
    msgs_b = hist_b.json()["messages"]
    for m in msgs_b:
        assert "Ritesh" not in m["content"]
