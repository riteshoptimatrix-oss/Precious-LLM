"""
Precious Edu LLM — Dataset Splitter Unit Tests
"""

import pytest
from app.dataset.splitter import DatasetSplitter
from app.dataset.models import DatasetRecord, RecordType


def test_deterministic_split_reproducibility():
    records = [
        DatasetRecord(record_id=f"r{i}", source_id="s1", type=RecordType.PLAIN_TEXT, text=f"Text item {i}")
        for i in range(20)
    ]

    splitter1 = DatasetSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    t1, v1, ts1 = splitter1.split(records)

    splitter2 = DatasetSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    t2, v2, ts2 = splitter2.split(records)

    assert [r.record_id for r in t1] == [r.record_id for r in t2]
    assert [r.record_id for r in v1] == [r.record_id for r in v2]
    assert [r.record_id for r in ts1] == [r.record_id for r in ts2]


def test_no_data_leakage_across_splits():
    records = [
        DatasetRecord(record_id=f"r{i}", source_id="s1", type=RecordType.PLAIN_TEXT, text=f"Text item {i}")
        for i in range(20)
    ]

    splitter = DatasetSplitter(seed=42)
    t, v, ts = splitter.split(records)

    train_ids = set(r.record_id for r in t)
    val_ids = set(r.record_id for r in v)
    test_ids = set(r.record_id for r in ts)

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
