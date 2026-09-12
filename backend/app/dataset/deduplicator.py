"""
Precious Edu LLM — Dataset Deduplicator

Executes exact duplicate detection using SHA-256 content hashing.
Prevents duplicate records from polluting training sets and leaking across splits.
"""

import hashlib
import logging
from typing import Set, Tuple, List
from app.dataset.models import DatasetRecord, RecordType

logger = logging.getLogger(__name__)


class DatasetDeduplicator:
    """
    Deduplicates DatasetRecords based on normalized content hashing.
    """

    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.duplicates_removed_count = 0

    def compute_content_hash(self, record: DatasetRecord) -> str:
        """
        Compute SHA-256 hash of record normalized content.
        """
        if record.type == RecordType.PLAIN_TEXT:
            content_str = record.text or ""
        elif record.type == RecordType.CONVERSATION and record.messages:
            msg_parts = [f"{m.role.lower()}:{m.content.strip()}" for m in record.messages]
            content_str = "||".join(msg_parts)
        else:
            content_str = ""

        return hashlib.sha256(content_str.encode("utf-8")).hexdigest()

    def process_record(self, record: DatasetRecord) -> Tuple[bool, str]:
        """
        Check if record is unique.

        Returns:
            Tuple of (is_unique: bool, content_hash: str).
        """
        content_hash = self.compute_content_hash(record)

        if content_hash in self.seen_hashes:
            self.duplicates_removed_count += 1
            return False, content_hash

        self.seen_hashes.add(content_hash)
        return True, content_hash

    def reset(self) -> None:
        self.seen_hashes.clear()
        self.duplicates_removed_count = 0
