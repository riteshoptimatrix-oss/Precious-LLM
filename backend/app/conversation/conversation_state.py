"""
Precious Edu LLM — Conversation State

Internal representation of active conversation state for a single turn.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ConversationState(BaseModel):
    """
    State object encapsulating loaded history, active memories, and session metadata.
    """
    session_id: str = Field(..., description="Target session ID")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Loaded message history")
    memories: Dict[str, str] = Field(default_factory=dict, description="Active session key-value memories")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Turn metadata and context budgeting info")
