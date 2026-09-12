"""
Precious AI — Website Crawler Models & Enums
"""

from enum import Enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CrawlStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    QUEUED = "QUEUED"
    FETCHING = "FETCHING"
    FETCHED = "FETCHED"
    PROCESSED = "PROCESSED"
    UNCHANGED = "UNCHANGED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    SKIPPED = "SKIPPED"
    REMOVED = "REMOVED"


class DiscoveredURL(BaseModel):
    url: str
    normalized_url: str
    depth: int = 0
    discovered_from: Optional[str] = None
    discovered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FetchedPage(BaseModel):
    url: str
    normalized_url: str
    canonical_url: str
    status_code: int
    content_type: str
    raw_html: str
    content_length: int
    fetch_duration_sec: float
    crawl_depth: int = 0
    error: Optional[str] = None
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CrawlRunSummary(BaseModel):
    crawl_id: str
    website_version: str
    base_url: str
    started_at: str
    completed_at: Optional[str] = None
    status: str = "RUNNING"
    pages_discovered: int = 0
    pages_fetched: int = 0
    pages_new: int = 0
    pages_changed: int = 0
    pages_unchanged: int = 0
    pages_removed: int = 0
    pages_failed: int = 0
    chunks_created: int = 0
    errors: List[Dict[str, Any]] = Field(default_factory=list)
