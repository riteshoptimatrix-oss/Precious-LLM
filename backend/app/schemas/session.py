"""
Precious Edu LLM — Session Request/Response Schemas
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class SessionCreate(BaseModel):
    """Schema for creating a new session."""
    title: Optional[str] = Field(default="New Conversation", description="Session title")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session metadata")

    @field_validator("title")
    def validate_title(cls, v: Optional[str]) -> str:
        if v is not None:
            v = v.strip()
            if not v:
                return "New Conversation"
            if len(v) > 200:
                raise ValueError("Session title cannot exceed 200 characters.")
        return v or "New Conversation"


class MessageResponse(BaseModel):
    """Schema for a single chat message response."""
    session_id: str
    role: str
    content: str
    created_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    """Schema for returning session details."""
    session_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionDetailResponse(SessionResponse):
    """Session details including message history."""
    messages: List[MessageResponse] = Field(default_factory=list)


class SessionListResponse(BaseModel):
    """List of chat sessions response."""
    sessions: List[SessionResponse]
    total: int
