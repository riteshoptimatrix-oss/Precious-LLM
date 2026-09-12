"""
Precious AI — Website Chunks MongoDB Repository
"""

import logging
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING

logger = logging.getLogger(__name__)


class WebsiteChunkRepository:
    """
    MongoDB repository for collection `website_chunks`.
    """

    COLLECTION_NAME = "website_chunks"

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db[self.COLLECTION_NAME]

    async def ensure_indexes(self) -> None:
        """Create search & lookup indexes."""
        try:
            await self.collection.create_index([("canonical_url", ASCENDING)], name="idx_chunk_url")
            await self.collection.create_index([("website_version", ASCENDING), ("active", ASCENDING)], name="idx_chunk_ver_active")
            await self.collection.create_index([("page_type", ASCENDING)], name="idx_chunk_page_type")
        except Exception as e:
            logger.warning(f"Index creation warning on website_chunks: {e}")

    async def save_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Saves chunk documents into collection."""
        if not chunks:
            return
        await self.ensure_indexes()
        await self.collection.insert_many(chunks)

    async def get_active_chunks(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Returns active chunks for search engine."""
        cursor = self.collection.find({"active": True})
        return await cursor.to_list(length=limit)

    async def set_active_version(self, website_version: str) -> None:
        """Deactivates old chunks and activates target version chunks."""
        await self.collection.update_many({}, {"$set": {"active": False}})
        await self.collection.update_many({"website_version": website_version}, {"$set": {"active": True}})
