"""
Precious Edu LLM — Common API Schemas

Shared Pydantic models used across multiple endpoints:
- Error responses
- Health check response
- Pagination
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ============================================
# Error Schemas
# ============================================

class ErrorDetail(BaseModel):
    """Structured error detail."""

    code: str = Field(
        ...,
        description="Machine-readable error code (e.g., VALIDATION_ERROR).",
        examples=["VALIDATION_ERROR"],
    )
    message: str = Field(
        ...,
        description="Human-readable error message.",
        examples=["Message exceeds maximum length of 2000 characters."],
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error context.",
    )


class ErrorResponse(BaseModel):
    """Standard error response envelope."""

    error: ErrorDetail


# ============================================
# Health Check Schemas
# ============================================

class ComponentHealth(BaseModel):
    """Health status of a single component."""

    status: str = Field(
        ...,
        description="Component status: connected, disconnected, loaded, not_loaded.",
    )
    type: Optional[str] = Field(None, description="Component type (e.g., mongodb).")
    message: Optional[str] = Field(None, description="Additional status info.")
    error: Optional[str] = Field(None, description="Error message if unhealthy.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(
        ...,
        description="Overall system status: ok, degraded, unhealthy.",
        examples=["ok"],
    )
    timestamp: str = Field(
        ...,
        description="ISO 8601 timestamp of the health check.",
    )
    version: str = Field(
        ...,
        description="Application version.",
        examples=["0.1.0"],
    )
    components: Dict[str, ComponentHealth] = Field(
        default_factory=dict,
        description="Health status of individual components.",
    )


# ============================================
# Pagination Schemas
# ============================================

class PaginationMeta(BaseModel):
    """Pagination metadata for list responses."""

    total: int = Field(..., description="Total number of items.")
    skip: int = Field(..., description="Number of items skipped.")
    limit: int = Field(..., description="Maximum items per page.")
    has_more: bool = Field(..., description="Whether more items exist.")
