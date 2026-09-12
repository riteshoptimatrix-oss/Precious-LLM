"""
Precious AI — Website Dataset Version Manager

Orchestrates atomic dataset activation and rollback across MongoDB collections.
Ensures website_pages, website_chunks, and website_knowledge_versions remain
consistent during version switches.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.website.repositories.page_repository import WebsitePageRepository
from app.website.repositories.chunk_repository import WebsiteChunkRepository
from app.website.repositories.version_repository import WebsiteVersionRepository

logger = logging.getLogger(__name__)


class WebsiteVersionManager:
    """
    Manages atomic website knowledge dataset versioning.

    Activation sequence (all-or-nothing):
    1. Deactivate all pages and chunks from other versions.
    2. Activate target version pages and chunks.
    3. Update version registry to mark target as active.

    Rollback is identical — just supply the previous version_id.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.page_repo = WebsitePageRepository(db)
        self.chunk_repo = WebsiteChunkRepository(db)
        self.version_repo = WebsiteVersionRepository(db)

    async def activate_version(self, website_version: str) -> Dict[str, Any]:
        """
        Atomically activates the specified website version.

        Args:
            website_version: Version identifier string (e.g., 'web-v1726123456')

        Returns:
            Dict with activation summary: version_id, pages_activated, chunks_activated, activated_at
        """
        # Verify target version exists in the registry
        versions = await self.version_repo.list_versions()
        version_ids = [v.get("website_version") for v in versions]

        if website_version not in version_ids:
            raise ValueError(
                f"Version '{website_version}' not found in registry. "
                f"Available versions: {version_ids}"
            )

        logger.info(f"Activating website version: {website_version}")

        # Step 1: Deactivate all and activate target in pages
        await self.page_repo.set_active_version(website_version)

        # Step 2: Deactivate all and activate target in chunks
        await self.chunk_repo.set_active_version(website_version)

        # Step 3: Update version registry
        await self.version_repo.activate_version(website_version)

        # Count activated documents for summary
        active_pages = await self.page_repo.get_active_pages()
        active_chunks = await self.chunk_repo.get_active_chunks(limit=10000)

        activated_at = datetime.now(timezone.utc).isoformat()

        summary = {
            "website_version": website_version,
            "pages_activated": len(active_pages),
            "chunks_activated": len(active_chunks),
            "activated_at": activated_at,
            "status": "ACTIVATED",
        }

        logger.info(
            f"Version '{website_version}' activated: "
            f"{summary['pages_activated']} pages, {summary['chunks_activated']} chunks"
        )
        return summary

    async def rollback_to_version(self, website_version: str) -> Dict[str, Any]:
        """
        Rolls back to a previous version. Semantically identical to activate_version
        but logs the operation as a rollback for audit clarity.

        Args:
            website_version: Version identifier to rollback to.

        Returns:
            Dict with rollback summary.
        """
        logger.warning(f"Rolling back to website version: {website_version}")
        summary = await self.activate_version(website_version)
        summary["status"] = "ROLLED_BACK"
        return summary

    async def get_active_version_info(self) -> Optional[Dict[str, Any]]:
        """
        Returns metadata about the currently active website version.

        Returns:
            Version document or None if no version is active.
        """
        return await self.version_repo.get_active_version()

    async def list_all_versions(self) -> List[Dict[str, Any]]:
        """
        Lists all recorded website knowledge versions in descending creation order.

        Returns:
            List of version metadata documents.
        """
        return await self.version_repo.list_versions()

    async def register_version(
        self,
        website_version: str,
        crawl_id: str,
        pages_count: int,
        chunks_count: int,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Saves a new version metadata record (does NOT activate it).

        Args:
            website_version: Unique version string.
            crawl_id: Associated crawl run ID.
            pages_count: Number of pages in this version.
            chunks_count: Number of chunks in this version.
            extra_meta: Optional additional metadata dict.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        version_doc: Dict[str, Any] = {
            "website_version": website_version,
            "crawl_id": crawl_id,
            "pages_count": pages_count,
            "chunks_count": chunks_count,
            "is_active": False,
            "status": "PENDING",
            "created_at": now_str,
            "activated_at": None,
        }
        if extra_meta:
            version_doc.update(extra_meta)

        await self.version_repo.save_version(version_doc)
        logger.info(
            f"Registered website version '{website_version}' "
            f"({pages_count} pages, {chunks_count} chunks)"
        )
