"""
Precious Edu LLM — Memory Service

Extracts and manages conversational memory:
- User name extraction from messages
- Entity extraction (topics, preferences)
- Memory persistence (short-term and long-term)
- Memory retrieval for context building

Initial implementation uses rule-based pattern matching.
Later phases will use model-based extraction.
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================
# Memory Extraction Patterns (Rule-Based)
# ============================================

# Patterns for extracting user name from messages
NAME_PATTERNS = [
    re.compile(r"(?:my name is|i'm|i am|call me)\s+(\w+)", re.IGNORECASE),
    re.compile(r"(?:this is)\s+(\w+)(?:\s+here)?", re.IGNORECASE),
]


class MemoryService:
    """
    Extracts and manages conversational memory.

    Memory types:
    - short_term: Scoped to current session (e.g., current topic)
    - long_term: Persists across sessions (e.g., user name, preferences)
    """

    def __init__(self, memory_repo=None):
        self.memory_repo = memory_repo

    async def extract_and_store(
        self,
        session_id: str,
        message: str,
        message_id: str,
    ) -> List[Dict]:
        """
        Extract memory facts from a user message and store them.

        Args:
            session_id: The current session ID.
            message: The user's message text.
            message_id: The message ID (for provenance tracking).

        Returns:
            List of extracted memory items.
        """
        # TODO (Phase 9): Implement
        raise NotImplementedError("MemoryService.extract_and_store — Phase 9")

    def extract_user_name(self, message: str) -> Optional[str]:
        """
        Extract user name from a message using pattern matching.

        Args:
            message: The user's message text.

        Returns:
            Extracted name or None.
        """
        for pattern in NAME_PATTERNS:
            match = pattern.search(message)
            if match:
                name = match.group(1).strip()
                # Basic validation: name should be alphabetic and reasonable length
                if name.isalpha() and 2 <= len(name) <= 30:
                    return name.capitalize()
        return None

    async def get_session_memory(self, session_id: str) -> Dict[str, str]:
        """
        Retrieve all memory items for a session.

        Args:
            session_id: The session ID.

        Returns:
            Dict mapping memory keys to values.
        """
        # TODO (Phase 9): Implement via memory_repo
        raise NotImplementedError("MemoryService.get_session_memory — Phase 9")
