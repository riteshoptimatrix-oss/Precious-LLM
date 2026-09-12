"""
Precious Edu LLM — Chat API Schemas

Pydantic models for chat request/response validation:
- ChatRequest: Incoming user message
- ChatResponse: Complete AI response
- ChatStreamEvent: Individual SSE event for streaming
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """
    Request body for sending a chat message.

    If session_id is omitted, a new session is created automatically.
    """

    session_id: Optional[str] = Field(
        None,
        description="Existing session ID. If omitted, a new session is created.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message text.",
        examples=["Hello, how are you?"],
    )

    @field_validator("message")
    @classmethod
    def validate_message_not_blank(cls, v: str) -> str:
        """Ensure the message is not just whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Message cannot be empty or only whitespace.")
        return stripped


class ChatMetadata(BaseModel):
    """Metadata about the generated response."""

    generation_time_ms: float = Field(
        ...,
        description="Time taken to generate the response in milliseconds.",
    )
    model_version: str = Field(
        ...,
        description="Version of the model that generated the response.",
    )
    token_count: int = Field(
        ...,
        description="Number of tokens in the generated response.",
    )


class ChatResponse(BaseModel):
    """
    Response body for a chat message.

    Contains the AI-generated response and metadata.
    """

    session_id: str = Field(
        ...,
        description="The session ID (new or existing).",
    )
    message_id: str = Field(
        ...,
        description="Unique ID of this message exchange.",
    )
    response: str = Field(
        ...,
        description="The AI-generated response text.",
    )
    turn_number: int = Field(
        ...,
        description="The turn number within the session.",
    )
    metadata: ChatMetadata = Field(
        ...,
        description="Generation metadata.",
    )


class ChatStreamEvent(BaseModel):
    """
    Individual event in an SSE stream.

    Each event contains either a partial token or a completion signal.
    """

    token: str = Field(
        "",
        description="Generated token text (partial response).",
    )
    done: bool = Field(
        False,
        description="Whether generation is complete.",
    )
    message_id: Optional[str] = Field(
        None,
        description="Message ID (only present in final event).",
    )
    metadata: Optional[ChatMetadata] = Field(
        None,
        description="Generation metadata (only present in final event).",
    )
