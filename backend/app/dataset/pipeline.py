"""
Precious Edu LLM — Dataset Pipeline Orchestrator

Main dataset processing orchestrator coordinating ingestion, normalization, validation,
quarantine, deduplication, quality filtering, statistics calculation, deterministic splitting,
atomic JSONL writing, manifest generation, and post-write verification.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.dataset.config import DatasetConfig, get_dataset_config
from app.dataset.models import DatasetRecord
from app.dataset.ingestor import DatasetIngestor
from app.dataset.normalizer import DatasetNormalizer
from app.dataset.validator import DatasetValidator
from app.dataset.quarantine import QuarantineManager
from app.dataset.deduplicator import DatasetDeduplicator
from app.dataset.quality import QualityFilter
from app.dataset.statistics import DatasetStatisticsCalculator
from app.dataset.splitter import DatasetSplitter
from app.dataset.writer import DatasetWriter
from app.dataset.manifest import DatasetManifestBuilder

logger = logging.getLogger(__name__)


class DatasetPipeline:
    """
    Main orchestrator for dataset engineering.
    """

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or get_dataset_config()
        self.config.ensure_directories()

        self.ingestor = DatasetIngestor()
        self.normalizer = DatasetNormalizer()
        self.validator = DatasetValidator()
        self.quarantine = QuarantineManager(self.config.QUARANTINE_DIR)
        self.deduplicator = DatasetDeduplicator()
        self.quality_filter = QualityFilter(
            min_characters=self.config.MIN_TEXT_CHARACTERS,
            max_characters=self.config.MAX_TEXT_CHARACTERS,
            max_repetition_ratio=self.config.MAX_REPETITION_RATIO
        )
        self.stats_calculator = DatasetStatisticsCalculator()
        self.splitter = DatasetSplitter(
            train_ratio=self.config.TRAIN_RATIO,
            val_ratio=self.config.VALIDATION_RATIO,
            test_ratio=self.config.TEST_RATIO,
            seed=self.config.SPLIT_SEED
        )
        self.writer = DatasetWriter()
        self.manifest_builder = DatasetManifestBuilder()

    def run(self) -> Dict[str, Any]:
        """
        Execute the 13-stage dataset pipeline.

        Returns:
            Dictionary of pipeline execution summary metrics.
        """
        logger.info(f"Starting Dataset Pipeline v{self.config.PIPELINE_VERSION}...")
        self.quarantine.clear()
        self.deduplicator.reset()

        # 1. Discover raw files
        raw_files = self.ingestor.discover_raw_files(self.config.RAW_DIR)
        logger.info(f"Discovered {len(raw_files)} raw files in {self.config.RAW_DIR}")

        records_read = 0
        valid_records: List[DatasetRecord] = []
        rejected_count = 0

        # 2-8. Ingest, Normalize, Validate, Quarantine, Deduplicate, Quality Filter
        for file_path in raw_files:
            for raw_record in self.ingestor.ingest_file(file_path):
                records_read += 1

                # Normalize
                normalized_rec = self.normalizer.normalize_record(raw_record)

                # Validate
                is_valid, val_reason = self.validator.validate_record(normalized_rec)
                if not is_valid:
                    self.quarantine.quarantine_record(normalized_rec, val_reason or "validation_failed")
                    rejected_count += 1
                    continue

                # Deduplicate
                if self.config.ENABLE_DEDUPLICATION:
                    is_unique, _ = self.deduplicator.process_record(normalized_rec)
                    if not is_unique:
                        self.quarantine.quarantine_record(normalized_rec, "exact_duplicate")
                        rejected_count += 1
                        continue

                # Quality Filter
                if self.config.ENABLE_QUALITY_FILTERS:
                    passed_q, q_reason = self.quality_filter.evaluate(normalized_rec)
                    if not passed_q:
                        self.quarantine.quarantine_record(normalized_rec, q_reason or "quality_failed")
                        rejected_count += 1
                        continue

                valid_records.append(normalized_rec)

        # 9. Calculate Overall Statistics
        overall_stats = self.stats_calculator.calculate_statistics(valid_records)

        # 10. Split Dataset Deterministically
        train_recs, val_recs, test_recs = self.splitter.split(valid_records)

        # 11. Write JSONL Outputs
        train_file = self.writer.write_jsonl(train_recs, self.config.TRAINING_DIR / "train.jsonl")
        val_file = self.writer.write_jsonl(val_recs, self.config.TRAINING_DIR / "validation.jsonl")
        test_file = self.writer.write_jsonl(test_recs, self.config.TRAINING_DIR / "test.jsonl")

        split_files = {
            "train": train_file,
            "validation": val_file,
            "test": test_file
        }
        split_counts = {
            "train": len(train_recs),
            "validation": len(val_recs),
            "test": len(test_recs)
        }

        # 12. Quality Report & Manifest Generation
        quality_report = {
            "records_read": records_read,
            "records_valid": len(valid_records),
            "records_rejected": rejected_count,
            "quarantined_count": self.quarantine.quarantined_count,
            "exact_duplicates_removed": self.deduplicator.duplicates_removed_count,
            "quality_filtered_empty": self.quality_filter.filtered_empty_count,
            "quality_filtered_length": self.quality_filter.filtered_length_count,
            "quality_filtered_repetition": self.quality_filter.filtered_repetition_count,
            "pass_rate_pct": round((len(valid_records) / records_read * 100), 2) if records_read > 0 else 0.0
        }

        config_snapshot = {
            "DATASET_VERSION": self.config.DATASET_VERSION,
            "PIPELINE_VERSION": self.config.PIPELINE_VERSION,
            "MIN_TEXT_CHARACTERS": self.config.MIN_TEXT_CHARACTERS,
            "MAX_TEXT_CHARACTERS": self.config.MAX_TEXT_CHARACTERS,
            "TRAIN_RATIO": self.config.TRAIN_RATIO,
            "VALIDATION_RATIO": self.config.VALIDATION_RATIO,
            "TEST_RATIO": self.config.TEST_RATIO,
            "SPLIT_SEED": self.config.SPLIT_SEED,
            "ENABLE_DEDUPLICATION": self.config.ENABLE_DEDUPLICATION,
            "ENABLE_QUALITY_FILTERS": self.config.ENABLE_QUALITY_FILTERS
        }

        self.manifest_builder.generate_manifest(
            output_dir=self.config.METADATA_DIR,
            dataset_version=self.config.DATASET_VERSION,
            pipeline_version=self.config.PIPELINE_VERSION,
            config_snapshot=config_snapshot,
            split_files=split_files,
            split_counts=split_counts,
            overall_stats=overall_stats,
            quality_report=quality_report
        )

        # 13. Output Verification
        self.verify_output(split_files, len(valid_records))

        logger.info(f"Dataset Pipeline completed successfully. Valid records: {len(valid_records)}")

        return {
            "records_read": records_read,
            "valid_records": len(valid_records),
            "train_records": len(train_recs),
            "validation_records": len(val_recs),
            "test_records": len(test_recs),
            "quality_report": quality_report
        }

    def verify_output(self, split_files: Dict[str, Path], expected_valid_count: int) -> bool:
        """
        Post-write verification validating lines and JSON integrity.
        """
        written_total = 0
        for name, path in split_files.items():
            if not path.exists():
                raise RuntimeError(f"Output verification failed: File {path} does not exist.")
            
            line_count = 0
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line_count += 1
                    line_str = line.strip()
                    if not line_str:
                        raise RuntimeError(f"Output verification failed: Empty line found at {name}:{idx}")
                    try:
                        json.loads(line_str)
                    except Exception as err:
                        raise RuntimeError(f"Output verification failed: Corrupted JSON at {name}:{idx} -> {err}")

            written_total += line_count

        if written_total != expected_valid_count:
            raise RuntimeError(
                f"Output verification failed: Written line sum ({written_total}) != "
                f"expected valid count ({expected_valid_count})"
            )

        logger.info(f"Output verification PASSED. Total written records: {written_total}")
        return True
