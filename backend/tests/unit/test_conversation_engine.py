"""
Precious Edu LLM — Conversation Engine Unit Tests
"""

import pytest
from app.core.exceptions import SessionNotFoundError
from app.conversation.engine import ConversationEngine
from app.services.session_service import SessionService


@pytest.mark.asyncio
async def test_conversation_engine_valid_turn(test_db):
    session_service = SessionService(test_db)
    session = await session_service.create_session("Test Engine Session")
    session_id = session["session_id"]

    engine = ConversationEngine(db=test_db)

    # Turn 1: Hello
    res1 = await engine.handle_message(session_id, "Hello")
    assert res1["session_id"] == session_id
    assert res1["role"] == "assistant"
    assert "Hello" in res1["response"]

    # Turn 2: Name introduction
    res2 = await engine.handle_message(session_id, "My name is Ritesh.")
    assert "Ritesh" in res2["response"]

    # Turn 3: Query name
    res3 = await engine.handle_message(session_id, "What is my name?")
    assert "Ritesh" in res3["response"]


@pytest.mark.asyncio
async def test_conversation_engine_invalid_session(test_db):
    engine = ConversationEngine(db=test_db)
    with pytest.raises(SessionNotFoundError):
        await engine.handle_message("nonexistent-session-id-999", "Hello")


@pytest.mark.asyncio
async def test_conversation_engine_generator_failure(test_db):
    session_service = SessionService(test_db)
    session = await session_service.create_session("Failing Generator Session")
    session_id = session["session_id"]

    class FailingGenerator:
        async def generate(self, context):
            raise RuntimeError("Generator internal error")

    engine = ConversationEngine(db=test_db, generator=FailingGenerator())

    with pytest.raises(RuntimeError):
        await engine.handle_message(session_id, "Hello")

    # Verify user message was saved despite generator failure
    messages = await engine.message_repo.get_by_session_id(session_id)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"
