"""
Precious Edu LLM — Dataset Validation Unit Tests
"""

import pytest
from app.dataset.validator import DatasetValidator
from app.dataset.models import DatasetRecord, RecordType, MessageRecord


def test_validate_empty_plain_text():
    validator = DatasetValidator()
    rec = DatasetRecord(record_id="r1", source_id="s1", type=RecordType.PLAIN_TEXT, text="")

    is_valid, reason = validator.validate_record(rec)
    assert is_valid is False
    assert reason == "empty_content"


def test_validate_invalid_role():
    validator = DatasetValidator()
    rec = DatasetRecord(
        record_id="r2",
        source_id="s1",
        type=RecordType.CONVERSATION,
        messages=[MessageRecord(role="banana", content="Hello")]
    )

    is_valid, reason = validator.validate_record(rec)
    assert is_valid is False
    assert "invalid_role_banana" in reason


def test_validate_valid_conversation():
    validator = DatasetValidator()
    rec = DatasetRecord(
        record_id="r3",
        source_id="s1",
        type=RecordType.CONVERSATION,
        messages=[
            MessageRecord(role="user", content="Hello"),
            MessageRecord(role="assistant", content="Hi!")
        ]
    )

    is_valid, reason = validator.validate_record(rec)
    assert is_valid is True
    assert reason is None
