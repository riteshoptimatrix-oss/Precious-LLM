"""
Precious Edu LLM — Response Generator Abstract Base Class

Abstract interface decoupling the ConversationEngine from specific model generation logic.
"""

from abc import ABC, abstractmethod
from typing import Any


class ResponseGenerator(ABC):
    """
    Abstract interface for model response generators.
    Decouples conversation context construction from text generation.
    """

    @abstractmethod
    async def generate(self, context: Any) -> str:
        """
        Generate a text response given a structured ConversationContext instance.

        Args:
            context: ConversationContext containing system_instructions, memories,
                     recent_messages, current_user_message, and metadata.

        Returns:
            Assistant response string.
        """
        pass
