"""
Precious Edu LLM — Dataset Quality Filtering Unit Tests
"""

import pytest
from app.dataset.quality import QualityFilter
from app.dataset.models import DatasetRecord, RecordType


def test_quality_filter_pathological_repetition():
    qfilter = QualityFilter()

    # Pathological repetition
    bad_rec = DatasetRecord(
        record_id="r1",
        source_id="s1",
        type=RecordType.PLAIN_TEXT,
        text="Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello"
    )

    passed, reason = qfilter.evaluate(bad_rec)
    assert passed is False
    assert reason == "quality_pathological_repetition"

    # Legitimate repetition should pass
    good_rec = DatasetRecord(
        record_id="r2",
        source_id="s1",
        type=RecordType.PLAIN_TEXT,
        text="Very very good job."
    )

    passed_good, _ = qfilter.evaluate(good_rec)
    assert passed_good is True
