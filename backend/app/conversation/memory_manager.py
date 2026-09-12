"""
Precious Edu LLM — Memory Manager

High-level manager for persistent conversation memory.
Interfaces with MemoryRepository and MemoryExtractor to manage session memories,
prevent duplicate memory entries, update existing memory keys, enforce session isolation,
and guarantee fault isolation for chat execution.
"""

import logging
from typing import Dict, List, Optional
from app.models.memory import MemoryModel
from app.repositories.memory_repository import MemoryRepository
from app.conversation.memory_extractor import MemoryExtractor
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manager governing memory lifecycle operations and session isolation.
    """

    def __init__(
        self,
        memory_repo: MemoryRepository,
        extractor: Optional[MemoryExtractor] = None,
        enabled: Optional[bool] = None,
    ):
        settings = get_settings()
        self.memory_repo = memory_repo
        self.extractor = extractor or MemoryExtractor()
        self.enabled = enabled if enabled is not None else settings.MEMORY_ENABLED

    async def get_memories_dict(self, session_id: str) -> Dict[str, str]:
        """
        Retrieve all active memories for a session as a key-value dict.

        Args:
            session_id: Target session ID.

        Returns:
            Dict mapping memory keys to memory values ({key: value}).
        """
        if not self.enabled:
            return {}

        try:
            memory_docs = await self.memory_repo.find_by_session(session_id)
            return {doc["key"]: doc["value"] for doc in memory_docs if "key" in doc and "value" in doc}
        except Exception as err:
            logger.error(f"Failed to fetch memories for session {session_id}: {err}", exc_info=True)
            return {}

    async def process_user_message(
        self,
        session_id: str,
        user_message: str,
        source_message_id: Optional[str] = None
    ) -> List[MemoryModel]:
        """
        Extract memories from incoming user message and update MemoryRepository.
        Fault-tolerant: memory processing errors are logged without throwing exceptions.

        Args:
            session_id: Session ID.
            user_message: Input message text.
            source_message_id: Message ID provenance.

        Returns:
            List of successfully extracted and saved MemoryModel objects.
        """
        if not self.enabled:
            return []

        try:
            extracted = self.extractor.extract_memories(
                session_id=session_id,
                user_message=user_message,
                source_message_id=source_message_id
            )

            saved_memories: List[MemoryModel] = []
            for mem in extracted:
                await self.memory_repo.create_or_update(mem)
                saved_memories.append(mem)
                logger.info(f"Persisted memory key '{mem.key}' = '{mem.value}' for session '{session_id}'")

            return saved_memories
        except Exception as err:
            logger.error(f"Error processing memory for session {session_id}: {err}", exc_info=True)
            return []

    async def set_memory(self, memory: MemoryModel) -> Optional[Dict]:
        """Directly insert or update a memory item."""
        if not self.enabled:
            return None
        try:
            return await self.memory_repo.create_or_update(memory)
        except Exception as err:
            logger.error(f"Failed to set memory key '{memory.key}': {err}", exc_info=True)
            return None

    async def delete_memory(self, session_id: str, key: str) -> bool:
        """Delete a memory item by session_id and key."""
        if not self.enabled:
            return False
        try:
            return await self.memory_repo.delete_by_key(session_id, key)
        except Exception as err:
            logger.error(f"Failed to delete memory key '{key}' for session {session_id}: {err}", exc_info=True)
            return False
