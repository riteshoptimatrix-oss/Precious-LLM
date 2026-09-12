"""
Precious Edu LLM — Quarantine Manager

Safely stores rejected and malformed records in data/intermediate/quarantine/
along with structural rejection reasons for debugging and auditability.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from app.dataset.models import DatasetRecord

logger = logging.getLogger(__name__)


class QuarantineManager:
    """
    Manages quarantining of rejected records.
    """

    def __init__(self, quarantine_dir: Path):
        self.quarantine_dir = quarantine_dir
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_file = self.quarantine_dir / "quarantined_records.jsonl"
        self.quarantined_count = 0

    def quarantine_record(self, record: DatasetRecord, reason: str) -> None:
        """
        Write a rejected record and its reason to the quarantine file.
        """
        payload: Dict[str, Any] = {
            "quarantine_reason": reason,
            "record_id": record.record_id,
            "source_id": record.source_id,
            "record_type": record.type.value if hasattr(record.type, "value") else record.type,
            "raw_record": record.model_dump(mode="json")
        }

        try:
            with open(self.quarantine_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.quarantined_count += 1
            logger.debug(f"Quarantined record {record.record_id} due to: {reason}")
        except Exception as err:
            logger.error(f"Failed to quarantine record {record.record_id}: {err}", exc_info=True)

    def clear(self) -> None:
        """Clear quarantine file before a fresh pipeline run."""
        if self.quarantine_file.exists():
            self.quarantine_file.unlink()
        self.quarantined_count = 0
