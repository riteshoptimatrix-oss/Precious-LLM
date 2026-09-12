"""
Precious Edu LLM — Project Records MongoDB Repository

Manages dataset storage, dataset versioning, version activation,
indexes, and retrieval operations in MongoDB collection `project_records`.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, TEXT

from app.knowledge.exceptions import KnowledgeRepositoryError
from app.knowledge.models import KnowledgeHealthReport, ProjectRecord

logger = logging.getLogger(__name__)


class ProjectRepository:
    """
    MongoDB repository for Project Records and Dataset Metadata.
    Collection: project_records
    Metadata Collection: project_dataset_metadata
    """

    COLLECTION_NAME = "project_records"
    METADATA_COLLECTION = "project_dataset_metadata"

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db[self.COLLECTION_NAME]
        self.meta_collection = db[self.METADATA_COLLECTION]

    async def ensure_indexes(self) -> None:
        """Create necessary indexes for search and versioning."""
        try:
            await self.collection.create_index(
                [("dataset_version", ASCENDING), ("project_id", ASCENDING)],
                name="idx_dataset_proj_id"
            )
            await self.collection.create_index(
                [("is_active", ASCENDING), ("project_id", ASCENDING)],
                name="idx_active_proj_id"
            )
            await self.collection.create_index(
                [("is_active", ASCENDING), ("project_name", ASCENDING)],
                name="idx_active_proj_name"
            )
            await self.collection.create_index(
                [("is_active", ASCENDING), ("client", ASCENDING)],
                name="idx_active_client"
            )
            await self.collection.create_index(
                [("is_active", ASCENDING), ("status", ASCENDING)],
                name="idx_active_status"
            )
            await self.collection.create_index(
                [("is_active", ASCENDING), ("manager", ASCENDING)],
                name="idx_active_manager"
            )
            logger.info("ProjectRepository indexes verified.")
        except Exception as e:
            logger.warning(f"Index check/creation warning: {e}")

    async def get_active_metadata(self) -> Optional[Dict[str, Any]]:
        """Fetch metadata for the currently active dataset version."""
        return await self.meta_collection.find_one({"is_active": True})

    async def get_metadata_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """Find dataset metadata by SHA-256 file hash."""
        return await self.meta_collection.find_one({"file_hash": file_hash})

    async def get_next_version_id(self) -> str:
        """Generates sequential version identifier like projects-v1, projects-v2."""
        count = await self.meta_collection.count_documents({})
        return f"projects-v{count + 1}"

    async def save_dataset(
        self,
        records: List[ProjectRecord],
        dataset_version: str,
        file_hash: str,
        source_file: str,
        activate: bool = True
    ) -> str:
        """
        Stores project records for a dataset version.
        Uses validation-before-activation workflow.
        """
        if not records:
            raise KnowledgeRepositoryError("Cannot save empty record set")

        await self.ensure_indexes()

        now_str = datetime.now(timezone.utc).isoformat()
        docs = []
        for r in records:
            d = r.model_dump(by_alias=False, exclude_none=True)
            d.pop("id", None)
            d.pop("_id", None)
            d["dataset_version"] = dataset_version
            d["is_active"] = False  # Inactive until validation complete
            docs.append(d)

        # 1. Insert records into collection
        try:
            await self.collection.insert_many(docs)
        except Exception as e:
            raise KnowledgeRepositoryError(f"Failed to insert project records: {e}") from e

        # 2. Save metadata entry
        meta_doc = {
            "dataset_version": dataset_version,
            "file_hash": file_hash,
            "source_file": source_file,
            "record_count": len(records),
            "is_active": False,
            "created_at": now_str,
            "activated_at": None,
        }
        await self.meta_collection.update_one(
            {"dataset_version": dataset_version},
            {"$set": meta_doc},
            upsert=True
        )

        # 3. Activate if requested
        if activate:
            await self.activate_dataset(dataset_version)

        return dataset_version

    async def activate_dataset(self, dataset_version: str) -> None:
        """
        Activates a specified dataset version and deactivates all others.
        """
        now_str = datetime.now(timezone.utc).isoformat()

        # Deactivate all project records across all versions
        await self.collection.update_many(
            {},
            {"$set": {"is_active": False}}
        )

        # Deactivate all dataset metadata entries across all versions
        await self.meta_collection.update_many(
            {},
            {"$set": {"is_active": False}}
        )

        # Activate target project records
        await self.collection.update_many(
            {"dataset_version": dataset_version},
            {"$set": {"is_active": True}}
        )

        # Activate target metadata
        await self.meta_collection.update_one(
            {"dataset_version": dataset_version},
            {"$set": {"is_active": True, "activated_at": now_str}}
        )

        logger.info(f"Dataset version {dataset_version} activated successfully.")


    async def get_health_status(self) -> KnowledgeHealthReport:
        """Generate health check status report."""
        active_meta = await self.get_active_metadata()
        if not active_meta:
            return KnowledgeHealthReport(
                status="EMPTY",
                active_dataset="",
                total_records=0,
            )

        active_version = active_meta.get("dataset_version", "")
        record_count = await self.collection.count_documents({"dataset_version": active_version, "is_active": True})

        return KnowledgeHealthReport(
            status="READY",
            active_dataset=active_version,
            total_records=record_count,
            last_ingested_at=active_meta.get("created_at"),
            source_file=active_meta.get("source_file"),
        )
