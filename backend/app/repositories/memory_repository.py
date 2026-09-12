"""
Precious Edu LLM — Memory Repository

Data access layer for memory collection in MongoDB.
Executes session-scoped CRUD operations for persistent conversation memory.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import DESCENDING

from app.db.collections import MEMORIES_COLLECTION
from app.models.memory import MemoryModel

logger = logging.getLogger(__name__)


class MemoryRepository:
    """Repository handling memory persistence in MongoDB."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db[MEMORIES_COLLECTION]

    async def create_or_update(self, memory: MemoryModel) -> Dict:
        """
        Upsert a memory document by (session_id, key).
        If memory key exists for session, updates value, updated_at, and metadata.
        """
        now = datetime.now(timezone.utc)
        doc = memory.to_dict()
        doc["updated_at"] = now

        result = await self.collection.find_one_and_update(
            {"session_id": memory.session_id, "key": memory.key},
            {
                "$set": {
                    "value": memory.value,
                    "type": memory.type.value if hasattr(memory.type, "value") else memory.type,
                    "source": memory.source,
                    "confidence": memory.confidence,
                    "updated_at": now,
                    "metadata": memory.metadata,
                },
                "$setOnInsert": {
                    "memory_id": memory.memory_id,
                    "session_id": memory.session_id,
                    "key": memory.key,
                    "created_at": memory.created_at,
                }
            },
            upsert=True,
            return_document=True
        )
        return result

    async def get_by_key(self, session_id: str, key: str) -> Optional[Dict]:
        """Fetch memory document by session_id and key."""
        return await self.collection.find_one({"session_id": session_id, "key": key})

    async def find_by_session(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Fetch all memories for a session ordered by updated_at DESC."""
        cursor = self.collection.find({"session_id": session_id}).sort("updated_at", DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)

    async def get_all_by_session(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Fetch all memories for a session (alias for find_by_session)."""
        return await self.find_by_session(session_id, limit=limit)

    async def delete_by_key(self, session_id: str, key: str) -> bool:
        """Delete a specific memory by session_id and key."""
        result = await self.collection.delete_one({"session_id": session_id, "key": key})
        return result.deleted_count > 0

    async def delete_all_by_session(self, session_id: str) -> int:
        """Delete all memories associated with a session ID."""
        result = await self.collection.delete_many({"session_id": session_id})
        return result.deleted_count
