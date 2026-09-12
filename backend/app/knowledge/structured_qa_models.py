"""
Precious Edu LLM — Structured Q&A Data Models

Defines Pydantic models for structured Q&A MongoDB documents,
candidates, search results, and ingestion statistics.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class StructuredQADocument(BaseModel):
    """
    MongoDB document schema for structured Q&A records.
    Source of truth is preserved in user_input, response, and intent.
    """
    id: Optional[str] = Field(None, alias="_id")
    intent: str = Field(..., description="Canonical intent identifier (e.g. fullform_peic, service_student_visas)")
    user_input: str = Field(..., description="Original user question / input text")
    response: str = Field(..., description="Original factual response / answer")
    normalized_input: str = Field(..., description="Cleaned, lowercased, punctuation-normalized question")
    tokens: List[str] = Field(default_factory=list, description="Extracted tokens from user_input")
    keywords: List[str] = Field(default_factory=list, description="Meaningful non-stop-word domain keywords")
    category: str = Field(..., description="Category / folder name (e.g. Services, FullForm, About_Us)")
    source: str = Field("structured_qa", description="Knowledge collection source tag")
    source_file: str = Field(..., description="Relative file path of the source JSON")
    active: bool = Field(True, description="Whether record is active for retrieval")
    record_hash: str = Field(..., description="Deterministic SHA256 hash of (intent, user_input, response)")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = ConfigDict(populate_by_name=True)


class StructuredQACandidate(BaseModel):
    """
    Scored candidate representation returned during retrieval.
    """
    record_id: str
    intent: str
    user_input: str
    response: str
    source_file: str
    category: str
    score: float
    score_breakdown: Dict[str, float] = Field(default_factory=dict)
    answerability: float = 1.0


class StructuredQAReport(BaseModel):
    """
    Summary report of an import run.
    """
    files_discovered: int = 0
    files_processed: int = 0
    records_discovered: int = 0
    records_imported: int = 0
    records_updated: int = 0
    duplicates_skipped: int = 0
    invalid_records: int = 0
    errors: List[str] = Field(default_factory=list)
    database: str = ""
    collection: str = ""
