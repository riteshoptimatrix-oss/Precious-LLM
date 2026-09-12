"""
Precious Edu LLM — Chat Service

High-level service endpoint layer interfacing between API routes and ConversationEngine.
Delegates chat turn orchestration to ConversationEngine while keeping route contracts intact.
"""

from typing import Any, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.conversation.engine import ConversationEngine
from app.llm.base import ResponseGenerator


class ChatService:
    """Service layer delegating chat turn orchestration to ConversationEngine."""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        generator: Optional[ResponseGenerator] = None,
        engine: Optional[ConversationEngine] = None,
    ):
        self.db = db
        self.engine = engine or ConversationEngine(db=db, generator=generator)

    async def process_chat(
        self,
        session_id: str,
        user_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delegate conversation processing to ConversationEngine.
        Returns exact Phase 2 compatible response structure:
        {
            "session_id": session_id,
            "response": response_text,
            "role": "assistant",
            "created_at": iso_timestamp_str,
            "metadata": metadata_dict
        }
        """
        return await self.engine.handle_message(
            session_id=session_id,
            user_message=user_message,
            metadata=metadata
        )
