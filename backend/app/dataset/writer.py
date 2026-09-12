"""
Precious Edu LLM — Dataset Writer

Safely writes canonical DatasetRecords to JSONL files (train.jsonl, validation.jsonl, test.jsonl)
using atomic temporary file swapping to prevent partial/corrupted writes.
"""

import json
import logging
from pathlib import Path
from typing import List
from app.dataset.models import DatasetRecord

logger = logging.getLogger(__name__)


class DatasetWriter:
    """
    Writes records to JSONL files atomically.
    """

    def write_jsonl(self, records: List[DatasetRecord], output_path: Path) -> Path:
        """
        Write DatasetRecord objects to target JSONL file atomically.

        Args:
            records: List of DatasetRecord instances.
            output_path: Target output path.

        Returns:
            Final Path of written file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")

        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                for rec in records:
                    export_dict = rec.to_export_dict()
                    f.write(json.dumps(export_dict, ensure_ascii=False) + "\n")

            # Atomic replace
            tmp_path.replace(output_path)
            logger.info(f"Wrote {len(records)} records atomically to {output_path}")
            return output_path

        except Exception as err:
            logger.error(f"Failed writing to {output_path}: {err}", exc_info=True)
            if tmp_path.exists():
                tmp_path.unlink()
            raise
