"""
Precious AI — Website Crawl Run Log MongoDB Repository
"""

import logging
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING

logger = logging.getLogger(__name__)


class CrawlRunRepository:
    """
    MongoDB repository for collection `website_crawl_runs`.
    """

    COLLECTION_NAME = "website_crawl_runs"

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db[self.COLLECTION_NAME]

    async def save_run(self, run_doc: Dict[str, Any]) -> None:
        """Saves crawl run summary."""
        await self.collection.update_one(
            {"crawl_id": run_doc["crawl_id"]},
            {"$set": run_doc},
            upsert=True
        )

    async def get_latest_run(self) -> Optional[Dict[str, Any]]:
        """Returns most recent crawl run."""
        cursor = self.collection.find({}).sort("started_at", -1).limit(1)
        runs = await cursor.to_list(length=1)
        return runs[0] if runs else None
