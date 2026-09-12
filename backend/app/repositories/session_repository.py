"""
Precious Edu LLM — Session Repository

Encapsulates MongoDB CRUD operations for sessions.
No business logic allowed here.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import DESCENDING

from app.db.collections import SESSIONS_COLLECTION
from app.models.session import SessionModel


class SessionRepository:
    """Repository for managing session persistence in MongoDB."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.collection = db[SESSIONS_COLLECTION]

    async def create(self, session: SessionModel) -> Dict[str, Any]:
        """Create a new session document."""
        doc = session.to_dict()
        await self.collection.insert_one(doc)
        doc.pop("_id", None)
        return doc

    async def get_by_session_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by session_id."""
        doc = await self.collection.find_one({"session_id": session_id}, {"_id": 0})
        return doc

    async def list_sessions(self, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
        """List sessions sorted by updated_at descending."""
        cursor = self.collection.find({}, {"_id": 0}).sort("updated_at", DESCENDING).skip(skip).limit(limit)
        return await cursor.to_list(length=limit)

    async def count_sessions(self) -> int:
        """Total count of sessions."""
        return await self.collection.count_documents({})

    async def update_timestamp(self, session_id: str) -> bool:
        """Update updated_at timestamp of a session."""
        result = await self.collection.update_one(
            {"session_id": session_id},
            {"$set": {"updated_at": datetime.now(timezone.utc)}}
        )
        return result.modified_count > 0

    async def delete(self, session_id: str) -> bool:
        """Delete session document by session_id."""
        result = await self.collection.delete_one({"session_id": session_id})
        return result.deleted_count > 0
