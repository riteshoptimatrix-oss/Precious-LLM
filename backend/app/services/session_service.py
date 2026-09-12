"""
Precious Edu LLM — Session Service

Coordinates SessionRepository and MessageRepository for session management operations.
"""

import uuid
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.exceptions import SessionNotFoundError
from app.models.session import SessionModel
from app.repositories.session_repository import SessionRepository
from app.repositories.message_repository import MessageRepository


class SessionService:
    """Service layer handling chat session business logic."""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.session_repo = SessionRepository(db)
        self.message_repo = MessageRepository(db)

    async def create_session(self, title: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a new chat session with a unique UUID session_id."""
        session_id = str(uuid.uuid4())
        session = SessionModel(
            session_id=session_id,
            title=title or "New Conversation",
            metadata=metadata or {}
        )
        return await self.session_repo.create(session)

    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Retrieve session details by session_id."""
        session = await self.session_repo.get_by_session_id(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        return session

    async def get_session_with_messages(self, session_id: str) -> Dict[str, Any]:
        """Retrieve session along with its full message history."""
        session = await self.get_session(session_id)
        messages = await self.message_repo.list_by_session_id(session_id)
        session["messages"] = messages
        return session

    async def list_sessions(self, limit: int = 50, skip: int = 0) -> Dict[str, Any]:
        """List sessions with total count."""
        sessions = await self.session_repo.list_sessions(limit=limit, skip=skip)
        total = await self.session_repo.count_sessions()
        return {"sessions": sessions, "total": total}

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its associated messages atomically/safely.
        """
        session = await self.session_repo.get_by_session_id(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # Delete messages first
        await self.message_repo.delete_by_session_id(session_id)
        # Delete session document
        return await self.session_repo.delete(session_id)
