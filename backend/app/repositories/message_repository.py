"""
Precious Edu LLM — Message Repository

Encapsulates MongoDB CRUD operations for messages.
No business logic allowed here.
"""

from typing import Any, Dict, List
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING

from app.db.collections import MESSAGES_COLLECTION
from app.models.message import MessageModel


class MessageRepository:
    """Repository for managing message persistence in MongoDB."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db[MESSAGES_COLLECTION]

    async def create(self, message: MessageModel) -> Dict[str, Any]:
        """Create a new message document."""
        doc = message.to_dict()
        await self.collection.insert_one(doc)
        doc.pop("_id", None)
        return doc

    async def list_by_session_id(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve latest messages for a session ordered chronologically."""
        cursor = self.collection.find({"session_id": session_id}, {"_id": 0}).sort("created_at", DESCENDING).limit(limit)
        messages = await cursor.to_list(length=limit)
        messages.reverse()
        return messages

    async def get_by_session_id(self, session_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Alias for list_by_session_id for context manager and engine compatibility."""
        return await self.list_by_session_id(session_id, limit=limit)

    async def delete_by_session_id(self, session_id: str) -> int:
        """Delete all messages associated with a session_id."""
        result = await self.collection.delete_many({"session_id": session_id})
        return result.deleted_count
