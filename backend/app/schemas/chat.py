"""
Precious Edu LLM — Chat Request/Response Schemas
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Schema for sending a chat message."""
    session_id: str = Field(..., description="Target session ID")
    message: str = Field(..., description="User message content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional request metadata")

    @field_validator("session_id")
    def validate_session_id(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("session_id cannot be empty or whitespace.")
        return v

    @field_validator("message")
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Message content cannot be empty.")
        if len(v) > 4000:
            raise ValueError("Message content exceeds maximum allowed length of 4000 characters.")
        return v


class ChatResponse(BaseModel):
    """Schema for chatbot response."""
    session_id: str
    response: str
    role: str = "assistant"
    created_at: str
    sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Knowledge source references used (structured Q&A or website chunks)."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
