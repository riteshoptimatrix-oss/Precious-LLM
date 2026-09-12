"""
Precious Edu LLM — Structured Q&A Idempotent Importer

Recursively scans backend/data/data/, normalizes records, computes deterministic
record_hash identifiers, and performs idempotent bulk upserts into the structured_qa
MongoDB collection.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import UpdateOne

from app.db.collections import STRUCTURED_QA_COLLECTION
from app.knowledge.structured_qa_models import StructuredQAReport
from app.knowledge.structured_qa_normalizer import StructuredQANormalizer

logger = logging.getLogger(__name__)


class StructuredQAImporter:
    """
    Idempotent importer for structured Q&A datasets.
    """

    def __init__(self, db: AsyncIOMotorDatabase, data_dir: Optional[str] = None):
        self.db = db
        self.collection_name = STRUCTURED_QA_COLLECTION
        if data_dir:
            self.data_dir = data_dir
        else:
            # Default to backend/data/data/ relative to project structure
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.data_dir = os.path.join(base_dir, "data", "data")

    def discover_files(self) -> List[str]:
        """Recursively discovers all JSON files in target data directory."""
        discovered = []
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return []

        for root, _, files in os.walk(self.data_dir):
            for f in sorted(files):
                if f.endswith(".json"):
                    discovered.append(os.path.join(root, f))
        return discovered

    def parse_file(self, filepath: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Safely parses JSON file and returns list of raw record dicts."""
        errors = []
        try:
            with open(filepath, "r", encoding="utf-8") as fp:
                content = json.load(fp)
            if isinstance(content, list):
                return content, errors
            elif isinstance(content, dict):
                return [content], errors
            else:
                errors.append(f"Unexpected top-level type {type(content).__name__} in {filepath}")
                return [], errors
        except Exception as e:
            errors.append(f"Failed to parse JSON file {filepath}: {str(e)}")
            return [], errors

    def normalize_record(self, raw_item: Any, source_file_rel: str, category: str) -> Optional[Dict[str, Any]]:
        """
        Validates and normalizes a single raw record.
        Supports both schemas:
          1. intent, user_input, response
          2. input, output (intent derived from category)
        """
        if not isinstance(raw_item, dict):
            return None

        # Schema detection
        user_input = raw_item.get("user_input") or raw_item.get("input")
        response = raw_item.get("response") or raw_item.get("output")
        intent = raw_item.get("intent") or category.lower()

        # Validation: required non-empty string fields
        if not user_input or not response or not str(user_input).strip() or not str(response).strip():
            return None

        clean_user_input = StructuredQANormalizer.clean_text(user_input)
        clean_response = StructuredQANormalizer.clean_text(response)
        clean_intent = StructuredQANormalizer.clean_text(intent).lower()

        normalized_input = StructuredQANormalizer.normalize_input(clean_user_input)
        tokens = StructuredQANormalizer.extract_tokens(clean_user_input)
        keywords = StructuredQANormalizer.extract_keywords(clean_user_input)
        record_hash = StructuredQANormalizer.generate_record_hash(clean_intent, clean_user_input, clean_response)

        now_iso = datetime.now(timezone.utc).isoformat()

        return {
            "intent": clean_intent,
            "user_input": clean_user_input,
            "response": clean_response,
            "normalized_input": normalized_input,
            "tokens": tokens,
            "keywords": keywords,
            "category": category,
            "source": "structured_qa",
            "source_file": source_file_rel,
            "active": True,
            "record_hash": record_hash,
            "created_at": now_iso,
            "updated_at": now_iso,
        }

    async def run_import(self, batch_size: int = 1000) -> StructuredQAReport:
        """
        Executes idempotent bulk import of all Q&A records into MongoDB.
        """
        report = StructuredQAReport(
            database=getattr(self.db, "name", "precious_edu_llm"),
            collection=self.collection_name,
        )

        files = self.discover_files()
        report.files_discovered = len(files)
        logger.info(f"Discovered {len(files)} JSON files in {self.data_dir}")

        collection = self.db[self.collection_name]
        # Ensure record_hash unique index exists for fast upserts
        try:
            await collection.create_index([("record_hash", 1)], unique=True, name="idx_sqa_record_hash_unique")
        except Exception as e:
            logger.debug(f"Index creation note: {e}")

        seen_hashes_in_batch = set()
        bulk_operations: List[UpdateOne] = []

        for filepath in files:
            rel_path = os.path.relpath(filepath, self.data_dir)
            category = os.path.basename(os.path.dirname(filepath)) or "General"
            raw_records, parse_errors = self.parse_file(filepath)

            if parse_errors:
                report.errors.extend(parse_errors)
                continue

            report.files_processed += 1
            file_records = len(raw_records)
            report.records_discovered += file_records

            for raw_item in raw_records:
                doc = self.normalize_record(raw_item, rel_path, category)
                if not doc:
                    report.invalid_records += 1
                    continue

                r_hash = doc["record_hash"]
                if r_hash in seen_hashes_in_batch:
                    report.duplicates_skipped += 1
                    continue
                seen_hashes_in_batch.add(r_hash)

                # Upsert operation based on deterministic record_hash
                update_doc = {
                    "$set": {
                        "intent": doc["intent"],
                        "user_input": doc["user_input"],
                        "response": doc["response"],
                        "normalized_input": doc["normalized_input"],
                        "tokens": doc["tokens"],
                        "keywords": doc["keywords"],
                        "category": doc["category"],
                        "source": doc["source"],
                        "source_file": doc["source_file"],
                        "active": doc["active"],
                        "updated_at": doc["updated_at"],
                    },
                    "$setOnInsert": {
                        "record_hash": doc["record_hash"],
                        "created_at": doc["created_at"],
                    }
                }
                bulk_operations.append(UpdateOne({"record_hash": r_hash}, update_doc, upsert=True))

                # Flush batch
                if len(bulk_operations) >= batch_size:
                    result = await collection.bulk_write(bulk_operations, ordered=False)
                    report.records_imported += result.upserted_count
                    report.records_updated += result.modified_count
                    bulk_operations = []

        # Flush any remaining operations
        if bulk_operations:
            result = await collection.bulk_write(bulk_operations, ordered=False)
            report.records_imported += result.upserted_count
            report.records_updated += result.modified_count
            bulk_operations = []

        logger.info(
            f"Import complete: {report.records_imported} inserted, "
            f"{report.records_updated} updated, {report.duplicates_skipped} duplicates skipped."
        )
        return report

    async def get_statistics(self) -> Dict[str, Any]:
        """Reports detailed dataset and collection statistics without running import."""
        files = self.discover_files()
        file_stats = []
        intent_counter: Dict[str, int] = {}
        total_discovered = 0
        total_unique = set()
        duplicate_count = 0
        invalid_count = 0

        for filepath in files:
            rel_path = os.path.relpath(filepath, self.data_dir)
            category = os.path.basename(os.path.dirname(filepath)) or "General"
            raw_records, _ = self.parse_file(filepath)
            file_intents = set()
            for r in raw_records:
                total_discovered += 1
                doc = self.normalize_record(r, rel_path, category)
                if not doc:
                    invalid_count += 1
                    continue
                r_hash = doc["record_hash"]
                intent = doc["intent"]
                file_intents.add(intent)
                intent_counter[intent] = intent_counter.get(intent, 0) + 1
                if r_hash in total_unique:
                    duplicate_count += 1
                else:
                    total_unique.add(r_hash)

            file_stats.append({
                "file": rel_path,
                "category": category,
                "records": len(raw_records),
                "unique_intents": len(file_intents),
            })

        db_count = await self.db[self.collection_name].count_documents({})

        return {
            "total_files": len(files),
            "total_records_discovered": total_discovered,
            "unique_records": len(total_unique),
            "duplicates": duplicate_count,
            "invalid_records": invalid_count,
            "unique_intents_count": len(intent_counter),
            "files_breakdown": file_stats,
            "top_intents": sorted(intent_counter.items(), key=lambda x: x[1], reverse=True)[:25],
            "mongodb_collection_count": db_count,
        }
