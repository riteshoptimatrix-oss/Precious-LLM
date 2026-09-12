"""
Precious Edu LLM — Context Manager Unit Tests
"""

import pytest
from app.models.message import MessageModel, MessageRole
from app.repositories.message_repository import MessageRepository
from app.conversation.context_manager import ContextManager


@pytest.mark.asyncio
async def test_context_manager_empty_history(test_db):
    repo = MessageRepository(test_db)
    cm = ContextManager(message_repo=repo)
    messages = await cm.get_recent_messages("empty-session")
    assert messages == []


@pytest.mark.asyncio
async def test_context_manager_max_messages_limit(test_db):
    repo = MessageRepository(test_db)
    session_id = "test-limit-session"

    # Create 15 messages
    for i in range(15):
        msg = MessageModel(
            session_id=session_id,
            role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"Message {i}"
        )
        await repo.create(msg)

    # CM with max_messages=6
    cm = ContextManager(message_repo=repo, max_messages=6, max_characters=10000)
    messages = await cm.get_recent_messages(session_id)

    assert len(messages) == 6
    assert messages[0]["content"] == "Message 9"
    assert messages[-1]["content"] == "Message 14"


@pytest.mark.asyncio
async def test_context_manager_max_characters_limit(test_db):
    repo = MessageRepository(test_db)
    session_id = "test-char-limit-session"

    # Add 4 long messages (each 100 chars)
    for i in range(4):
        msg = MessageModel(
            session_id=session_id,
            role=MessageRole.USER,
            content=f"Turn {i}: " + ("x" * 90)
        )
        await repo.create(msg)

    # CM with max_characters=250 (should only fit last 2 messages)
    cm = ContextManager(message_repo=repo, max_messages=10, max_characters=250)
    messages = await cm.get_recent_messages(session_id)

    assert len(messages) <= 2
    assert messages[-1]["content"].startswith("Turn 3:")
