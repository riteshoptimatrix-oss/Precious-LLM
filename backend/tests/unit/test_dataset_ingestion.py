"""
Precious Edu LLM — Dataset Ingestion Unit Tests
"""

import pytest
from pathlib import Path
from app.dataset.ingestor import DatasetIngestor
from app.dataset.models import RecordType


def test_ingest_txt_file(tmp_path: Path):
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello world.\n\nSecond line.\n", encoding="utf-8")

    ingestor = DatasetIngestor()
    records = list(ingestor.ingest_file(txt_file, source_id="test_txt"))

    assert len(records) == 2
    assert records[0].type == RecordType.PLAIN_TEXT
    assert records[0].text == "Hello world."
    assert records[1].text == "Second line."


def test_ingest_jsonl_file(tmp_path: Path):
    jsonl_file = tmp_path / "convs.jsonl"
    jsonl_file.write_text(
        '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}\n'
        '{"text": "Plain text in jsonl"}\n',
        encoding="utf-8"
    )

    ingestor = DatasetIngestor()
    records = list(ingestor.ingest_file(jsonl_file, source_id="test_jsonl"))

    assert len(records) == 2
    assert records[0].type == RecordType.CONVERSATION
    assert len(records[0].messages) == 2
    assert records[1].type == RecordType.PLAIN_TEXT
    assert records[1].text == "Plain text in jsonl"
