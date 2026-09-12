"""
Precious Edu LLM — Domain Schema Module

Pydantic models for domain training records, provenance tracking,
and quarantine reports supporting 7 distinct training example types.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class DomainExampleType(str, Enum):
    INSTRUCTION_RESPONSE = "instruction_response"
    TERMINOLOGY = "terminology"
    SERVICE = "service"
    CONVERSATION = "conversation"
    CLARIFICATION = "clarification"
    AMBIGUITY = "ambiguity"
    NEGATIVE_EXAMPLE = "negative_example"


class DomainProvenance(BaseModel):
    """Source provenance tracking for domain dataset records."""
    source_id: str = Field(..., description="Unique source identifier")
    source_name: str = Field(..., description="Name of source document or manual")
    source_type: str = Field("business_document", description="Type of source document")
    version: str = Field("1.0.0", description="Document or dataset version")
    language: str = Field("en", description="ISO language code")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DomainRecord(BaseModel):
    """Schema for a domain training example."""
    record_id: str = Field(..., description="Unique identifier for the domain record")
    type: DomainExampleType = Field(..., description="Example type (1 to 7)")
    category: str = Field("general", description="Taxonomy category key")
    instruction: Optional[str] = Field(None, description="Instruction prompt text")
    response: Optional[str] = Field(None, description="Target assistant response text")
    term: Optional[str] = Field(None, description="Terminology term if type==terminology")
    definition: Optional[str] = Field(None, description="Term definition if type==terminology")
    service_name: Optional[str] = Field(None, description="Service name if type==service")
    messages: Optional[List[Dict[str, str]]] = Field(None, description="Conversational turns ([{role, content}])")
    provenance: DomainProvenance = Field(..., description="Provenance metadata")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata")

    model_config = ConfigDict(use_enum_values=True)


class QuarantinedRecord(BaseModel):
    """Container for invalid domain records failed during validation."""
    record_id: str
    reason: str
    errors: List[str]
    raw_data: Dict[str, Any]
    quarantined_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
