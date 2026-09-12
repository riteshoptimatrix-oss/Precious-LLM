"""
Precious Edu LLM — Structured Q&A MongoDB Repository

Provides fast indexed queries against the structured_qa collection.
Ensures candidates are fetched efficiently without loading the entire collection into memory.
"""

import logging
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.collections import STRUCTURED_QA_COLLECTION

logger = logging.getLogger(__name__)


class StructuredQARepository:
    """
    Data access layer for structured_qa collection in MongoDB.
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db[STRUCTURED_QA_COLLECTION]

    async def get_exact_match(self, normalized_input: str) -> Optional[Dict[str, Any]]:
        """Finds record with an exact match on normalized_input."""
        if not normalized_input:
            return None
        return await self.collection.find_one({
            "normalized_input": normalized_input,
            "active": True
        })

    async def get_candidates(
        self,
        normalized_query: str,
        keywords: List[str],
        intent: Optional[str] = None,
        limit: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top candidate records matching intent or domain keywords.
        Ensures exact matches and intent-aligned records are always prioritized.
        """
        candidates: List[Dict[str, Any]] = []
        seen_ids = set()

        # 1. Always fetch exact matches on normalized_input first
        if normalized_query:
            cursor_exact = self.collection.find({
                "normalized_input": normalized_query,
                "active": True
            }).limit(20)
            async for doc in cursor_exact:
                doc_id = str(doc["_id"])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    candidates.append(doc)

        # 2. Fetch candidates matching predicted intent
        if intent and intent not in ("unknown", "greeting"):
            cursor_intent = self.collection.find({
                "intent": intent,
                "active": True
            }).limit(50)
            async for doc in cursor_intent:
                doc_id = str(doc["_id"])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    candidates.append(doc)

        # 3. Fetch candidates matching domain keywords up to the limit
        if keywords and len(candidates) < limit:
            remaining = limit - len(candidates)
            cursor_kw = self.collection.find({
                "keywords": {"$in": keywords},
                "active": True
            }).limit(remaining * 2)
            async for doc in cursor_kw:
                doc_id = str(doc["_id"])
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    candidates.append(doc)
                if len(candidates) >= limit:
                    break

        return candidates

    async def get_records_by_intent(self, intent: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetches active records matching a specific intent."""
        cursor = self.collection.find({"intent": intent, "active": True}).limit(limit)
        return await cursor.to_list(length=limit)

    async def get_health_status(self) -> Dict[str, Any]:
        """Provides status and counts for health endpoint."""
        try:
            total_active = await self.collection.count_documents({"active": True})
            unique_intents = await self.collection.distinct("intent", {"active": True})
            unique_categories = await self.collection.distinct("category", {"active": True})
            return {
                "is_ready": total_active > 0,
                "collection": STRUCTURED_QA_COLLECTION,
                "total_active_records": total_active,
                "unique_intents_count": len(unique_intents),
                "categories": unique_categories,
            }
        except Exception as e:
            logger.error(f"Structured QA health check failed: {e}")
            return {
                "is_ready": False,
                "error": str(e)
            }
