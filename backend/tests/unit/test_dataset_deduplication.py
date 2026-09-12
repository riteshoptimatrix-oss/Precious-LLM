"""
Precious Edu LLM — Dataset Deduplication Unit Tests
"""

import pytest
from app.dataset.deduplicator import DatasetDeduplicator
from app.dataset.models import DatasetRecord, RecordType


def test_exact_duplicate_detection():
    dedup = DatasetDeduplicator()

    rec1 = DatasetRecord(record_id="r1", source_id="s1", type=RecordType.PLAIN_TEXT, text="Hello world.")
    rec2 = DatasetRecord(record_id="r2", source_id="s1", type=RecordType.PLAIN_TEXT, text="Hello world.")

    is_unique1, hash1 = dedup.process_record(rec1)
    is_unique2, hash2 = dedup.process_record(rec2)

    assert is_unique1 is True
    assert is_unique2 is False
    assert hash1 == hash2
    assert dedup.duplicates_removed_count == 1
