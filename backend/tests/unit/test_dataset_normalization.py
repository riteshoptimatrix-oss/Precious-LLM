"""
Precious Edu LLM — Dataset Normalization Unit Tests
"""

import pytest
from app.dataset.normalizer import DatasetNormalizer
from app.dataset.models import DatasetRecord, RecordType


def test_normalization_whitespace_and_line_endings():
    normalizer = DatasetNormalizer()
    raw = "  Hello   world.  \r\n\r\n\r\n\r\nNext line.  "
    normalized = normalizer.normalize_text(raw)

    assert normalized == "Hello world.\n\nNext line."


def test_normalization_unicode_preservation():
    normalizer = DatasetNormalizer()
    raw = "Hello भारत 你好 こんにちは é ñ What?"
    normalized = normalizer.normalize_text(raw)

    assert normalized == "Hello भारत 你好 こんにちは é ñ What?"
    assert "?" in normalized  # Ensures question mark is preserved


def test_normalize_record_plain_text():
    normalizer = DatasetNormalizer()
    rec = DatasetRecord(
        record_id="r1",
        source_id="s1",
        type=RecordType.PLAIN_TEXT,
        text="  Some   text.  \n\n\n"
    )
    norm_rec = normalizer.normalize_record(rec)
    assert norm_rec.text == "Some text."
