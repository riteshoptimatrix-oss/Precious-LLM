"""
Precious AI — Website Pages MongoDB Repository
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING

logger = logging.getLogger(__name__)


class WebsitePageRepository:
    """
    MongoDB repository for collection `website_pages`.
    """

    COLLECTION_NAME = "website_pages"

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db[self.COLLECTION_NAME]

    async def ensure_indexes(self) -> None:
        """Create search & lookup indexes."""
        try:
            await self.collection.create_index([("canonical_url", ASCENDING)], name="idx_canonical_url")
            await self.collection.create_index([("content_hash", ASCENDING)], name="idx_content_hash")
            await self.collection.create_index([("website_version", ASCENDING), ("active", ASCENDING)], name="idx_ver_active")
            await self.collection.create_index([("page_type", ASCENDING)], name="idx_page_type")
        except Exception as e:
            logger.warning(f"Index creation warning on website_pages: {e}")

    async def save_pages(self, pages: List[Dict[str, Any]]) -> None:
        """Saves page documents into collection."""
        if not pages:
            return
        await self.ensure_indexes()
        await self.collection.insert_many(pages)

    async def get_active_pages(self) -> List[Dict[str, Any]]:
        """Returns active pages."""
        cursor = self.collection.find({"active": True})
        return await cursor.to_list(length=1000)

    async def get_by_canonical_url(self, canonical_url: str) -> Optional[Dict[str, Any]]:
        """Lookup active page by canonical URL."""
        return await self.collection.find_one({"canonical_url": canonical_url, "active": True})

    async def set_active_version(self, website_version: str) -> None:
        """Deactivates old pages and activates target version pages."""
        await self.collection.update_many({}, {"$set": {"active": False}})
        await self.collection.update_many({"website_version": website_version}, {"$set": {"active": True}})
