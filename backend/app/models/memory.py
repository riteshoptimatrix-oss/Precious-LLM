"""
Precious Edu LLM — Memory Database Model

Represents an extracted memory fact document in MongoDB.
Supported memory types: user_fact, preference, conversation_fact.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    USER_FACT = "user_fact"
    PREFERENCE = "preference"
    CONVERSATION_FACT = "conversation_fact"


class MemoryModel(BaseModel):
    """
    Memory document representation.
    """
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str = Field(..., description="Target session ID")
    key: str = Field(..., description="Memory key (e.g. 'user_name')")
    value: str = Field(..., description="Memory value (e.g. 'Ritesh')")
    type: MemoryType = Field(default=MemoryType.USER_FACT, description="Category of memory")
    source: str = Field(default="conversation", description="Origin of memory extraction")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provenance or context metadata")

    def to_dict(self) -> Dict[str, Any]:
        """Convert model instance to MongoDB document dict."""
        data = self.model_dump()
        data["type"] = self.type.value
        return data
