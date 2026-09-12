"""
Precious Edu LLM — Deterministic Project Knowledge Search

Executes deterministic local MongoDB queries against the active dataset version.
Supports exact ID match, prefix search, case-insensitive text search,
and structured multi-field filtering. Caps maximum returned records.
"""

import logging
import re
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.knowledge.exceptions import KnowledgeSearchError
from app.knowledge.models import ProjectRecord, SearchResult, StructuredProjectQuery

logger = logging.getLogger(__name__)

PROJECT_SEARCH_MAX_RESULTS = 5


class ProjectKnowledgeSearch:
    """
    Local MongoDB search engine for project records.
    """

    def __init__(self, db: AsyncIOMotorDatabase, max_results: int = PROJECT_SEARCH_MAX_RESULTS):
        self.db = db
        self.collection = db["project_records"]
        self.meta_collection = db["project_dataset_metadata"]
        self.max_results = max_results

    async def get_active_dataset_version(self) -> Optional[str]:
        """Fetch current active dataset version ID."""
        active_doc = await self.meta_collection.find_one({"is_active": True})
        if active_doc:
            return active_doc.get("dataset_version")
        return None

    async def search(
        self,
        query: StructuredProjectQuery,
        dataset_version: Optional[str] = None
    ) -> SearchResult:
        """
        Executes a validated StructuredProjectQuery against project records.
        """
        active_version = dataset_version or await self.get_active_dataset_version()
        if not active_version:
            return SearchResult(
                query=query,
                records=[],
                total_found=0,
                dataset_version=""
            )

        # Base mongo filter for active version
        mongo_filter: Dict[str, Any] = {
            "dataset_version": active_version,
            "is_active": True,
        }

        filters = query.filters or {}

        # 1. Exact project_id match
        if "project_id" in filters and filters["project_id"]:
            pid = str(filters["project_id"]).strip()
            mongo_filter["project_id"] = {"$regex": f"^{re.escape(pid)}$", "$options": "i"}

        # 2. Project name (exact, prefix, or substring)
        elif "project_name" in filters and filters["project_name"]:
            pname = str(filters["project_name"]).strip()
            mongo_filter["project_name"] = {"$regex": re.escape(pname), "$options": "i"}

        # 3. Client filter
        if "client" in filters and filters["client"]:
            cname = str(filters["client"]).strip()
            mongo_filter["client"] = {"$regex": re.escape(cname), "$options": "i"}

        # 4. Status filter
        if "status" in filters and filters["status"]:
            sname = str(filters["status"]).strip()
            mongo_filter["status"] = {"$regex": f"^{re.escape(sname)}$", "$options": "i"}

        # 5. Manager filter
        if "manager" in filters and filters["manager"]:
            mname = str(filters["manager"]).strip()
            mongo_filter["manager"] = {"$regex": re.escape(mname), "$options": "i"}

        # 6. General free text query
        if "query" in filters and filters["query"]:
            qtext = str(filters["query"]).strip()
            pattern = re.escape(qtext)
            text_or_conditions = [
                {"project_id": {"$regex": pattern, "$options": "i"}},
                {"project_name": {"$regex": pattern, "$options": "i"}},
                {"client": {"$regex": pattern, "$options": "i"}},
                {"manager": {"$regex": pattern, "$options": "i"}},
                {"description": {"$regex": pattern, "$options": "i"}},
            ]
            if "$or" in mongo_filter:
                mongo_filter["$and"] = [{"$or": mongo_filter["$or"]}, {"$or": text_or_conditions}]
                del mongo_filter["$or"]
            else:
                mongo_filter["$or"] = text_or_conditions

        limit = min(query.limit or self.max_results, self.max_results)

        try:
            cursor = self.collection.find(mongo_filter).limit(limit)
            docs = await cursor.to_list(length=limit)
            total_count = len(docs)

            records = []
            for d in docs:
                d["_id"] = str(d["_id"])
                records.append(ProjectRecord(**d))

            return SearchResult(
                query=query,
                records=records,
                total_found=total_count,
                dataset_version=active_version
            )
        except Exception as e:
            logger.error(f"Error executing project knowledge search: {e}")
            raise KnowledgeSearchError(f"Search failed: {e}") from e
