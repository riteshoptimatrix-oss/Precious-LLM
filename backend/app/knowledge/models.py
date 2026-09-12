"""
Precious Edu LLM — Project Knowledge Engine Data Models

Pydantic models representing canonical project records, provenance metadata,
ingestion reports, structured queries, and health status.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class SourceMetadata(BaseModel):
    """Provenance tracking for a project record."""
    file: str = Field(..., description="Source Excel file name or relative path")
    sheet: str = Field(..., description="Source sheet name within the workbook")
    row: int = Field(..., description="1-indexed row number in the sheet")
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    file_hash: str = Field("", description="SHA-256 hash of the source Excel file")
    dataset_version: str = Field("", description="Target dataset version identifier")


class ProjectRecord(BaseModel):
    """Canonical structure for project records stored in MongoDB."""
    id: Optional[str] = Field(None, alias="_id")
    project_id: str = Field(..., description="Unique project identifier (e.g., P001)")
    project_name: str = Field(..., description="Human-readable project name")
    client: str = Field("Unknown", description="Client name or organization")
    status: str = Field("Active", description="Project operational status (e.g. In Progress, Hold, Active)")
    manager: str = Field("Unassigned", description="Project manager name")
    start_date: Optional[str] = Field(None, description="ISO format start date YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="ISO format end date YYYY-MM-DD")
    description: str = Field("", description="Detailed project description or summary")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional arbitrary key-value metadata fields")
    source: SourceMetadata = Field(..., description="Source provenance details")
    dataset_version: str = Field(..., description="Dataset version (e.g., projects-v1)")
    is_active: bool = Field(True, description="Whether this record belongs to the active dataset version")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    model_config = ConfigDict(populate_by_name=True)


class IngestionReport(BaseModel):
    """Summary report produced after inspecting or ingesting an Excel workbook."""
    filename: str
    file_size_bytes: int
    file_hash: str
    dataset_version: str
    sheet_count: int
    sheet_names: List[str]
    sheet_details: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    total_rows_parsed: int = 0
    valid_records_count: int = 0
    invalid_records_count: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StructuredProjectQuery(BaseModel):
    """Validated structured query model for knowledge repository queries."""
    entity: str = Field("project", description="Target entity type")
    operation: str = Field("get", description="Query operation type (get, search, list)")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query key-value filter parameters")
    fields: List[str] = Field(default_factory=list, description="Target field subset requested")
    limit: int = Field(5, description="Maximum results limit")


class SearchResult(BaseModel):
    """Container for query results passed to context builder."""
    query: StructuredProjectQuery
    records: List[ProjectRecord]
    total_found: int
    dataset_version: str


class KnowledgeHealthReport(BaseModel):
    """Status info report for administrators."""
    status: str = Field("READY", description="Knowledge engine status (READY, EMPTY, ERROR)")
    active_dataset: str = Field("", description="Current active dataset version ID")
    total_records: int = Field(0, description="Total active project records")
    last_ingested_at: Optional[str] = Field(None, description="ISO timestamp of last dataset ingestion")
    source_file: Optional[str] = Field(None, description="Primary source file for active dataset")
