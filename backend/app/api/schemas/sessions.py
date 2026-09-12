"""
Precious Edu LLM — Session API Schemas

Pydantic models for session management:
- SessionCreateRequest
- SessionResponse
- SessionListResponse
- SessionDetailResponse (includes messages)
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from app.api.schemas.common import PaginationMeta


class SessionCreateRequest(BaseModel):
    """Request body for creating a new session."""

    title: Optional[str] = Field(
        None,
        max_length=200,
        description="Optional session title. Auto-generated if omitted.",
    )


class MessageResponse(BaseModel):
    """A single message within a session."""

    message_id: str = Field(..., description="Unique message ID.")
    role: str = Field(
        ...,
        description="Message role: 'user', 'assistant', or 'system'.",
        examples=["user"],
    )
    content: str = Field(..., description="Message text content.")
    turn_number: int = Field(..., description="Turn number within the session.")
    created_at: datetime = Field(..., description="When the message was created.")


class SessionResponse(BaseModel):
    """Response body for a single session (without messages)."""

    session_id: str = Field(..., description="Unique session ID.")
    title: Optional[str] = Field(None, description="Session title.")
    created_at: datetime = Field(..., description="When the session was created.")
    updated_at: datetime = Field(..., description="When the session was last updated.")
    message_count: int = Field(0, description="Number of messages in the session.")
    status: str = Field("active", description="Session status: active, archived, deleted.")


class SessionListResponse(BaseModel):
    """Response body for listing sessions."""

    sessions: List[SessionResponse] = Field(
        default_factory=list,
        description="List of sessions.",
    )
    pagination: PaginationMeta = Field(
        ...,
        description="Pagination metadata.",
    )


class SessionDetailResponse(BaseModel):
    """Response body for a session with its messages."""

    session_id: str = Field(..., description="Unique session ID.")
    title: Optional[str] = Field(None, description="Session title.")
    created_at: datetime = Field(..., description="When the session was created.")
    updated_at: datetime = Field(..., description="When the session was last updated.")
    status: str = Field(..., description="Session status.")
    messages: List[MessageResponse] = Field(
        default_factory=list,
        description="All messages in the session, ordered by turn number.",
    )
