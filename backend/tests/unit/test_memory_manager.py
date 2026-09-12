"""
Precious Edu LLM — Memory Manager Unit Tests
"""

import pytest
from app.models.memory import MemoryModel, MemoryType
from app.repositories.memory_repository import MemoryRepository
from app.conversation.memory_manager import MemoryManager


@pytest.mark.asyncio
async def test_memory_manager_crud(test_db):
    repo = MemoryRepository(test_db)
    mm = MemoryManager(memory_repo=repo)
    session_id = "mm-crud-session"

    # Process message: "My name is Ritesh."
    saved = await mm.process_user_message(session_id, "My name is Ritesh.")
    assert len(saved) == 1
    assert saved[0].key == "user_name"
    assert saved[0].value == "Ritesh"

    # Get memories dict
    mem_dict = await mm.get_memories_dict(session_id)
    assert mem_dict == {"user_name": "Ritesh"}

    # Update name: "Actually, my name is Raj."
    saved_update = await mm.process_user_message(session_id, "Actually, my name is Raj.")
    assert len(saved_update) == 1
    assert saved_update[0].value == "Raj"

    # Verify memory dict updated, single key remains
    mem_dict_updated = await mm.get_memories_dict(session_id)
    assert mem_dict_updated == {"user_name": "Raj"}

    # Delete memory
    deleted = await mm.delete_memory(session_id, "user_name")
    assert deleted is True
    assert await mm.get_memories_dict(session_id) == {}


@pytest.mark.asyncio
async def test_memory_manager_session_isolation(test_db):
    repo = MemoryRepository(test_db)
    mm = MemoryManager(memory_repo=repo)

    session_a = "session-a-123"
    session_b = "session-b-456"

    # Store memory for Session A
    await mm.process_user_message(session_a, "My name is Ritesh.")
    mem_a = await mm.get_memories_dict(session_a)
    assert mem_a == {"user_name": "Ritesh"}

    # Session B should have no memory
    mem_b = await mm.get_memories_dict(session_b)
    assert mem_b == {}


@pytest.mark.asyncio
async def test_memory_manager_fault_isolation(test_db):
    class FailingMemoryRepo:
        async def find_by_session(self, session_id):
            raise RuntimeError("Database connection lost")

        async def create_or_update(self, memory):
            raise RuntimeError("Database write error")

    failing_repo = FailingMemoryRepo()
    mm = MemoryManager(memory_repo=failing_repo)

    # Process message should not throw exception
    saved = await mm.process_user_message("failing-session", "My name is Ritesh.")
    assert saved == []

    # Get memories dict should return empty dict safely
    mem_dict = await mm.get_memories_dict("failing-session")
    assert mem_dict == {}
