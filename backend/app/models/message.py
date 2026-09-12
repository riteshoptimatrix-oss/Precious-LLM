"""
Precious Edu LLM — Message Database Model

Represents a single chat message document in MongoDB.
Roles supported: user, assistant, system.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageModel(BaseModel):
    """
    Message document representation.
    """
    session_id: str = Field(..., description="Target session ID")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Message text content")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible metadata (tokens, model version, generation params, latency)"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to MongoDB document dict."""
        data = self.model_dump()
        data["role"] = self.role.value
        return data
