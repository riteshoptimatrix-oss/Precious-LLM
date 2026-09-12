"""
Precious Edu LLM — Context Manager

Manages short-term conversation context loading from MessageRepository.
Enforces max message limits and max character budgeting on active context
while leaving full database history intact in MongoDB.
"""

import logging
from typing import Any, Dict, List, Optional
from app.repositories.message_repository import MessageRepository
from app.core.config import get_settings

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Handles context history windowing and character budgeting.
    """

    def __init__(
        self,
        message_repo: MessageRepository,
        max_messages: Optional[int] = None,
        max_characters: Optional[int] = None,
    ):
        settings = get_settings()
        self.message_repo = message_repo
        self.max_messages = max_messages or settings.CONTEXT_MAX_MESSAGES
        self.max_characters = max_characters or settings.CONTEXT_MAX_CHARACTERS

    async def get_recent_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Load recent message history for session within max_messages and max_characters constraints.

        Args:
            session_id: Target session ID.

        Returns:
            Chronologically ordered list of message dicts ({role, content, created_at}).
        """
        # Load messages from repository (most recent up to max_messages * 2)
        fetch_limit = self.max_messages * 2
        raw_messages = await self.message_repo.get_by_session_id(session_id, limit=fetch_limit)

        if not raw_messages:
            return []

        # Sort raw messages chronologically (created_at ASC)
        # Note: MessageRepository returns latest messages, but we ensure chronological order
        raw_messages.sort(key=lambda m: m.get("created_at") if m.get("created_at") else "")

        # Truncate by max_messages
        messages = raw_messages[-self.max_messages:] if len(raw_messages) > self.max_messages else raw_messages

        # Enforce max_characters budgeting from newest to oldest
        budgeted_messages: List[Dict[str, Any]] = []
        current_chars = 0

        for msg in reversed(messages):
            content = msg.get("content", "")
            msg_len = len(content)

            if current_chars + msg_len > self.max_characters and budgeted_messages:
                # Character budget exceeded; stop including older messages
                break

            budgeted_messages.append({
                "role": msg.get("role"),
                "content": content,
                "created_at": str(msg.get("created_at")) if msg.get("created_at") else None
            })
            current_chars += msg_len

        # Re-sort to chronological order (oldest first, newest last)
        budgeted_messages.reverse()
        return budgeted_messages
