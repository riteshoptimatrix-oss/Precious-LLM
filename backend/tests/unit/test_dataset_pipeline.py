"""
Precious Edu LLM — Dataset Pipeline End-to-End Unit Tests
"""

import pytest
from pathlib import Path
from app.dataset.pipeline import DatasetPipeline
from app.dataset.config import DatasetConfig


def test_pipeline_end_to_end_run(tmp_path: Path):
    # Setup test data directory
    raw_dir = tmp_path / "data" / "raw" / "conversations"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_file = raw_dir / "test.jsonl"
    raw_file.write_text(
        '{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}\n'
        '{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}\n'  # duplicate
        '{"text": ""}\n',  # empty / invalid
        encoding="utf-8"
    )

    config = DatasetConfig(
        PROJECT_ROOT=tmp_path,
        DATA_DIR=tmp_path / "data",
        RAW_DIR=tmp_path / "data" / "raw",
        INTERMEDIATE_DIR=tmp_path / "data" / "intermediate",
        QUARANTINE_DIR=tmp_path / "data" / "intermediate" / "quarantine",
        CLEANED_DIR=tmp_path / "data" / "cleaned",
        PROCESSED_DIR=tmp_path / "data" / "processed",
        TRAINING_DIR=tmp_path / "data" / "training",
        EVALUATION_DIR=tmp_path / "data" / "evaluation",
        METADATA_DIR=tmp_path / "data" / "metadata"
    )

    pipeline = DatasetPipeline(config=config)
    results = pipeline.run()

    assert results["records_read"] == 3
    assert results["valid_records"] == 1
    assert results["quality_report"]["quarantined_count"] == 2
    assert (config.TRAINING_DIR / "train.jsonl").exists()
    assert (config.METADATA_DIR / "dataset_manifest.json").exists()
