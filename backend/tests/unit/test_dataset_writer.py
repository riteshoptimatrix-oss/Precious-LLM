"""
Precious Edu LLM — Dataset Writer Unit Tests
"""

import json
import pytest
from pathlib import Path
from app.dataset.writer import DatasetWriter
from app.dataset.models import DatasetRecord, RecordType


def test_atomic_writer(tmp_path: Path):
    output_file = tmp_path / "out.jsonl"
    records = [
        DatasetRecord(record_id="r1", source_id="s1", type=RecordType.PLAIN_TEXT, text="Hello world")
    ]

    writer = DatasetWriter()
    writer.write_jsonl(records, output_file)

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["text"] == "Hello world"
