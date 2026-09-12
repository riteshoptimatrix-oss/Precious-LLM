"""
Precious Edu LLM — Dataset Manifest & Quality Report Builder

Generates dataset metadata artifacts:
- dataset_manifest.json: Complete dataset provenance, versions, SHA-256 checksums, and config snapshot
- statistics.json: Empirical record, conversation, turn, and character counts
- quality_report.json: Filtering statistics, rejection metrics, and quarantine summaries
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from app.dataset.models import DatasetRecord

logger = logging.getLogger(__name__)


class DatasetManifestBuilder:
    """
    Builds dataset manifest, statistics, quality report, and file checksums.
    """

    def compute_file_sha256(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of a file."""
        if not file_path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)

        return sha256.hexdigest()

    def generate_manifest(
        self,
        output_dir: Path,
        dataset_version: str,
        pipeline_version: str,
        config_snapshot: Dict[str, Any],
        split_files: Dict[str, Path],
        split_counts: Dict[str, int],
        overall_stats: Dict[str, Any],
        quality_report: Dict[str, Any]
    ) -> Path:
        """
        Generate dataset_manifest.json, statistics.json, and quality_report.json.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Compute split checksums
        checksums = {}
        for split_name, file_path in split_files.items():
            if file_path and file_path.exists():
                checksums[split_name] = {
                    "file_name": file_path.name,
                    "sha256": self.compute_file_sha256(file_path),
                    "size_bytes": file_path.stat().st_size,
                    "record_count": split_counts.get(split_name, 0)
                }

        # 1. Manifest
        manifest_payload = {
            "dataset_version": dataset_version,
            "pipeline_version": pipeline_version,
            "created_at": timestamp,
            "config_snapshot": config_snapshot,
            "splits": checksums,
            "overall_summary": {
                "total_records": overall_stats.get("total_records", 0),
                "total_conversations": overall_stats.get("total_conversations", 0),
                "total_plain_texts": overall_stats.get("total_plain_texts", 0),
                "total_messages": overall_stats.get("total_messages", 0),
                "total_characters": overall_stats.get("character_stats", {}).get("total_characters", 0)
            }
        }

        manifest_path = output_dir / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n")

        # 2. Statistics
        stats_path = output_dir / "statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(overall_stats, indent=2, ensure_ascii=False) + "\n")

        # 3. Quality Report
        quality_path = output_dir / "quality_report.json"
        with open(quality_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(quality_report, indent=2, ensure_ascii=False) + "\n")

        logger.info(f"Generated manifest, statistics, and quality report in {output_dir}")
        return manifest_path
