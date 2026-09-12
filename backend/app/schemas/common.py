"""
Precious Edu LLM — Common API Schemas

Shared response schemas: Health, Error, and Database health responses.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(..., json_schema_extra={"example": "ok"})
    service: str = Field(default="Precious AI")
    database: str = Field(..., json_schema_extra={"example": "connected"})
    version: str = Field(default="0.1.0")
    timestamp: Optional[str] = None
    components: Optional[Dict[str, Any]] = None


class DatabaseHealthResponse(BaseModel):
    """Database specific health response schema."""
    status: str = Field(..., json_schema_extra={"example": "ok"})
    database: str = Field(..., json_schema_extra={"example": "connected"})
    database_name: str = Field(..., json_schema_extra={"example": "precious_ai"})


class ErrorResponse(BaseModel):
    """Standard error response structure across all endpoints."""
    error: str = Field(..., description="Short error code or title")
    message: str = Field(..., description="Human readable error message")
    status_code: int = Field(..., description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Optional error metadata")
