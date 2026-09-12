"""
Precious AI — Website Knowledge Version MongoDB Repository
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING

logger = logging.getLogger(__name__)


class WebsiteVersionRepository:
    """
    MongoDB repository for collection `website_knowledge_versions`.
    """

    COLLECTION_NAME = "website_knowledge_versions"

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db[self.COLLECTION_NAME]

    async def get_active_version(self) -> Optional[Dict[str, Any]]:
        """Returns metadata of currently active version."""
        return await self.collection.find_one({"is_active": True, "status": "ACTIVE"})

    async def save_version(self, version_doc: Dict[str, Any]) -> None:
        """Saves version metadata."""
        await self.collection.update_one(
            {"website_version": version_doc["website_version"]},
            {"$set": version_doc},
            upsert=True
        )

    async def activate_version(self, website_version: str) -> None:
        """Sets target version as active and deactivates others."""
        now_str = datetime.now(timezone.utc).isoformat()
        await self.collection.update_many({}, {"$set": {"is_active": False}})
        await self.collection.update_one(
            {"website_version": website_version},
            {"$set": {"is_active": True, "status": "ACTIVE", "activated_at": now_str}}
        )

    async def list_versions(self) -> List[Dict[str, Any]]:
        """Lists all recorded website knowledge versions."""
        cursor = self.collection.find({}).sort("created_at", -1)
        return await cursor.to_list(length=100)
