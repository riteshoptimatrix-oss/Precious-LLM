"""
Precious Edu LLM — Conversation Manager

Manages conversation state for a single session:
- Loads conversation history from the database
- Tracks turn numbers
- Manages the conversation window (last N turns)
- Coordinates with ContextBuilder to create model prompts
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages the state of a single conversation session.

    Responsibilities:
    - Load conversation history from the database
    - Maintain a sliding window of recent turns
    - Track current turn number
    - Provide conversation state to the ContextBuilder
    """

    def __init__(
        self,
        session_id: str,
        message_repo=None,
        max_history_turns: int = 10,
    ):
        self.session_id = session_id
        self.message_repo = message_repo
        self.max_history_turns = max_history_turns
        self.history: List[Dict] = []
        self.current_turn: int = 0

    async def load_history(self) -> List[Dict]:
        """
        Load conversation history from the database.

        Returns:
            List of message dicts ordered by turn number.
        """
        # TODO (Phase 9): Implement via message_repo
        raise NotImplementedError("ConversationManager.load_history — Phase 9")

    def get_recent_history(self, max_turns: int | None = None) -> List[Dict]:
        """
        Get the most recent conversation turns.

        Args:
            max_turns: Override for max history turns. Uses default if None.

        Returns:
            List of recent message dicts (user + assistant pairs).
        """
        turns = max_turns or self.max_history_turns
        return self.history[-turns * 2:]  # Each turn = user + assistant message

    def add_message(self, role: str, content: str) -> int:
        """
        Add a message to the conversation history.

        Args:
            role: Message role ('user', 'assistant', 'system').
            content: Message text.

        Returns:
            The turn number of the added message.
        """
        if role == "user":
            self.current_turn += 1

        self.history.append({
            "role": role,
            "content": content,
            "turn_number": self.current_turn,
        })

        return self.current_turn
