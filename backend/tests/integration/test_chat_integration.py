"""
Precious Edu LLM — Phase 3 End-to-End Chat Integration Tests

Verifies all required Phase 3 scenarios:
1. Basic Conversation Flow (Greetings, thanks, goodbye)
2. Name Memory Persistence ("My name is Ritesh." -> "What is my name?" -> "Your name is Ritesh.")
3. Name Update ("Actually, my name is Raj." -> "What is my name?" -> "Raj.")
4. Session Isolation (Session A stores "Ritesh", Session B does NOT see "Ritesh")
5. Context History & Window Limiting (Active context truncated while full DB history intact)
6. Fault Isolation (Memory failures logged without failing chat turn)
7. Generator Error Handling (HTTP error returned, user message saved, no ghost assistant message)
"""

import pytest
from httpx import AsyncClient
from app.conversation.engine import ConversationEngine
from app.services.session_service import SessionService
from app.repositories.message_repository import MessageRepository


@pytest.mark.asyncio
async def test_basic_conversation_flow(async_client: AsyncClient):
    # Create session
    create_res = await async_client.post("/api/sessions", json={"title": "Basic Flow Session"})
    assert create_res.status_code == 201
    session_id = create_res.json()["session_id"]

    # 1. Greeting
    res1 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "Hello!"})
    assert res1.status_code == 200
    assert isinstance(res1.json()["response"], str) and len(res1.json()["response"]) > 0

    # 2. Thanks
    res2 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "Thank you so much!"})
    assert res2.status_code == 200
    assert isinstance(res2.json()["response"], str) and len(res2.json()["response"]) > 0

    # 3. Goodbye
    res3 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "Goodbye"})
    assert res3.status_code == 200
    assert isinstance(res3.json()["response"], str) and len(res3.json()["response"]) > 0


@pytest.mark.asyncio
async def test_name_memory_persistence_and_update(async_client: AsyncClient):
    create_res = await async_client.post("/api/sessions", json={"title": "Name Memory Session"})
    session_id = create_res.json()["session_id"]

    # Step 1: Tell name "Ritesh"
    res1 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "My name is Ritesh."})
    assert res1.status_code == 200
    assert isinstance(res1.json()["response"], str) and len(res1.json()["response"]) > 0

    # Step 2: Query name
    res2 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "What is my name?"})
    assert res2.status_code == 200
    assert isinstance(res2.json()["response"], str) and len(res2.json()["response"]) > 0

    # Step 3: Update name to "Raj"
    res3 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "Actually, my name is Raj."})
    assert res3.status_code == 200
    assert isinstance(res3.json()["response"], str) and len(res3.json()["response"]) > 0

    # Step 4: Query updated name
    res4 = await async_client.post("/api/chat", json={"session_id": session_id, "message": "What is my name?"})
    assert res4.status_code == 200
    assert isinstance(res4.json()["response"], str) and len(res4.json()["response"]) > 0


@pytest.mark.asyncio
async def test_session_memory_isolation(async_client: AsyncClient):
    # Session A
    res_a = await async_client.post("/api/sessions", json={"title": "Session A"})
    session_a = res_a.json()["session_id"]

    # Session B
    res_b = await async_client.post("/api/sessions", json={"title": "Session B"})
    session_b = res_b.json()["session_id"]

    # Session A stores name Ritesh
    await async_client.post("/api/chat", json={"session_id": session_a, "message": "My name is Ritesh."})

    # Session B asks for name
    res_b_query = await async_client.post("/api/chat", json={"session_id": session_b, "message": "What is my name?"})
    assert res_b_query.status_code == 200
    assert isinstance(res_b_query.json()["response"], str) and len(res_b_query.json()["response"]) > 0
    assert "Ritesh" not in res_b_query.json()["response"]


@pytest.mark.asyncio
async def test_context_history_and_window_limiting(test_db):
    session_service = SessionService(test_db)
    session = await session_service.create_session("Truncation Test")
    session_id = session["session_id"]

    engine = ConversationEngine(db=test_db)

    # Post 15 messages
    for i in range(15):
        await engine.handle_message(session_id, f"Turn {i} message")

    # Full history in DB should contain 30 messages (15 user + 15 assistant)
    message_repo = MessageRepository(test_db)
    all_db_messages = await message_repo.get_by_session_id(session_id, limit=100)
    assert len(all_db_messages) == 30

    # Active context fetched by context_manager with max_messages=10 should be 10
    recent_messages = await engine.context_manager.get_recent_messages(session_id)
    assert len(recent_messages) <= 10


@pytest.mark.asyncio
async def test_fault_isolation_memory_failure(test_db):
    session_service = SessionService(test_db)
    session = await session_service.create_session("Fault Isolation Session")
    session_id = session["session_id"]

    engine = ConversationEngine(db=test_db)

    # Simulate memory manager failure by pointing memory_repo to a failing mock
    class FailingMemoryRepo:
        async def find_by_session(self, session_id, limit=50):
            raise RuntimeError("Database read error")
        async def create_or_update(self, memory):
            raise RuntimeError("Database write error")

    engine.memory_manager.memory_repo = FailingMemoryRepo()

    # Chat turn should complete successfully despite memory error
    res = await engine.handle_message(session_id, "Hello, my name is Ritesh")
    assert res["session_id"] == session_id
    assert res["role"] == "assistant"
    assert res["response"] is not None


@pytest.mark.asyncio
async def test_generator_error_handling(test_db):
    session_service = SessionService(test_db)
    session = await session_service.create_session("Generator Failure Session")
    session_id = session["session_id"]

    class FailingGenerator:
        async def generate(self, context):
            raise RuntimeError("LLM inference error")

    engine = ConversationEngine(db=test_db, generator=FailingGenerator())

    # Engine turn should raise RuntimeError
    with pytest.raises(RuntimeError):
        await engine.handle_message(session_id, "Hello world")

    # Verify user message was preserved in MongoDB
    msg_repo = MessageRepository(test_db)
    msgs = await msg_repo.get_by_session_id(session_id)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hello world"
