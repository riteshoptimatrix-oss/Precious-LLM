"""
Precious Edu LLM — Canonical Dataset Models

Defines the internal canonical structure for all ingested dataset records.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecordType(str, Enum):
    PLAIN_TEXT = "plain_text"
    CONVERSATION = "conversation"
    INSTRUCTION_RESPONSE = "instruction_response"


class SourceProvenance(BaseModel):
    """
    Provenance tracking metadata for dataset sources.
    """
    source_id: str = Field(..., description="Unique source identifier")
    source_name: str = Field(..., description="Descriptive name of the raw dataset source")
    source_type: str = Field(default="text", description="Type of raw data (text, conversations, external)")
    license: str = Field(default="unknown", description="Legal usage license")
    origin: str = Field(default="local", description="URL or file system path of origin")
    language: str = Field(default="en", description="Primary language code")
    version: str = Field(default="1.0.0", description="Source dataset version")
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump()
        data["ingested_at"] = self.ingested_at.isoformat()
        return data


class MessageRecord(BaseModel):
    """
    Single message in a conversational record.
    """
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Text content of the message")


class DatasetRecord(BaseModel):
    """
    Internal canonical representation of a dataset record.
    Every ingested item (text, conversation, instruction) is normalized to this structure.
    """
    record_id: str = Field(..., description="Deterministic record hash/ID")
    source_id: str = Field(..., description="ID of source provenance")
    type: RecordType = Field(default=RecordType.PLAIN_TEXT, description="Category of record")
    language: str = Field(default="en", description="Language code")
    text: Optional[str] = Field(default=None, description="Plain text content (for PLAIN_TEXT)")
    messages: Optional[List[MessageRecord]] = Field(default=None, description="List of messages (for CONVERSATION)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata")
    provenance: Optional[SourceProvenance] = Field(default=None, description="Source provenance details")

    def to_export_dict(self) -> Dict[str, Any]:
        """Convert canonical record into clean JSONL export format."""
        if self.type == RecordType.CONVERSATION and self.messages:
            return {
                "messages": [{"role": m.role, "content": m.content} for m in self.messages]
            }
        else:
            return {
                "text": self.text or ""
            }
