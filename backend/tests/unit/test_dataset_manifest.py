"""
Precious Edu LLM — Dataset Manifest Builder Unit Tests
"""

import json
import pytest
from pathlib import Path
from app.dataset.manifest import DatasetManifestBuilder


def test_manifest_generation(tmp_path: Path):
    split_file = tmp_path / "train.jsonl"
    split_file.write_text('{"text": "sample"}\n', encoding="utf-8")

    builder = DatasetManifestBuilder()
    manifest_path = builder.generate_manifest(
        output_dir=tmp_path,
        dataset_version="0.1.0",
        pipeline_version="0.1.0",
        config_snapshot={"seed": 42},
        split_files={"train": split_file},
        split_counts={"train": 1},
        overall_stats={"total_records": 1},
        quality_report={"pass_rate": 100.0}
    )

    assert manifest_path.exists()
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["dataset_version"] == "0.1.0"
    assert "train" in data["splits"]
    assert "sha256" in data["splits"]["train"]
    assert len(data["splits"]["train"]["sha256"]) == 64
