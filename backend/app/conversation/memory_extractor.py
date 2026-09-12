"""
Precious Edu LLM — Memory Extractor

Deterministic pattern-matching extractor for Phase 3.
Identifies facts (e.g. user_name) from incoming user messages.
"""

import re
import logging
from typing import Dict, List, Optional
from app.models.memory import MemoryModel, MemoryType

logger = logging.getLogger(__name__)

# Pattern regexes for user name extraction
NAME_PATTERNS = [
    re.compile(r"(?:my name is|i'm|i am|call me)\s+([A-Za-z]+)", re.IGNORECASE),
    re.compile(r"^([A-Za-z]+)\s+here$", re.IGNORECASE),
]

# Excluded generic words to prevent false positive name extractions
NAME_EXCLUSIONS = {
    "hello", "hi", "hey", "sorry", "thanks", "thank", "yes", "no", "ok",
    "okay", "sure", "please", "help", "good", "great", "fine", "bye", "goodbye",
    "actually", "well", "now", "here", "there", "a", "an", "the"
}


class MemoryExtractor:
    """
    Extracts structured memory facts from user messages.
    """

    def extract_memories(
        self,
        session_id: str,
        user_message: str,
        source_message_id: Optional[str] = None
    ) -> List[MemoryModel]:
        """
        Scan message text for extractions and return MemoryModel instances.

        Args:
            session_id: Target session ID.
            user_message: Message text content.
            source_message_id: Optional message ID provenance.

        Returns:
            List of extracted MemoryModel objects.
        """
        memories: List[MemoryModel] = []
        if not user_message:
            return memories

        # 1. User Name Extraction
        name = self.extract_user_name(user_message)
        if name:
            memories.append(MemoryModel(
                session_id=session_id,
                key="user_name",
                value=name,
                type=MemoryType.USER_FACT,
                source="conversation",
                confidence=1.0,
                metadata={"source_message_id": source_message_id} if source_message_id else {}
            ))

        return memories

    def extract_user_name(self, text: str) -> Optional[str]:
        """
        Extract user name using deterministic patterns.

        Args:
            text: Raw message text.

        Returns:
            Capitalized name string if matched, else None.
        """
        if not text:
            return None

        for pattern in NAME_PATTERNS:
            match = pattern.search(text.strip())
            if match:
                candidate = match.group(1).strip()
                if (
                    candidate.isalpha()
                    and 2 <= len(candidate) <= 30
                    and candidate.lower() not in NAME_EXCLUSIONS
                ):
                    return candidate.capitalize()

        return None
