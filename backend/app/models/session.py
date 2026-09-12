"""
Precious Edu LLM — Session Database Model

Represents a chat conversation session document stored in MongoDB.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class SessionModel(BaseModel):
    """
    Session document representation.
    """
    session_id: str = Field(..., description="Unique application-level session identifier")
    title: str = Field(default="New Conversation", description="Human-readable title of the chat session")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to MongoDB document dict."""
        data = self.model_dump()
        return data
