"""
Precious Edu LLM — Unit Tests for Mapping, Normalization, and Validation

Tests header mapping aliases, safe whitespace/date normalizations,
required field checks, duplicate ID detection, and error reporting.
"""

import pytest
from app.knowledge.mapper import ColumnMapper
from app.knowledge.normalizer import DataNormalizer
from app.knowledge.validator import RecordValidator


def test_column_mapper_aliases():
    mapper = ColumnMapper()

    raw_row = {
        "Project Code": "P001",
        "Project Title": "Project Alpha",
        "Client Name": "ABC Ltd",
        "PM": "John Doe",
        "State": "In Progress",
        "Budget": 50000,
    }

    canonical, attributes = mapper.map_row(raw_row)

    assert canonical["project_id"] == "P001"
    assert canonical["project_name"] == "Project Alpha"
    assert canonical["client"] == "ABC Ltd"
    assert canonical["manager"] == "John Doe"
    assert canonical["status"] == "In Progress"
    assert attributes["Budget"] == 50000


def test_data_normalizer_dates_and_whitespace():
    normalizer = DataNormalizer()

    canonical = {
        "project_id": "  P001  ",
        "project_name": " Project Alpha ",
        "start_date": "2026-01-10",
        "end_date": "15/06/2026",
    }
    attributes = {" Priority ": " High "}

    norm_canonical, norm_attr = normalizer.normalize_record(canonical, attributes)

    assert norm_canonical["project_id"] == "P001"
    assert norm_canonical["project_name"] == "Project Alpha"
    assert norm_canonical["start_date"] == "2026-01-10"
    assert norm_canonical["end_date"] == "2026-06-15"
    assert norm_attr["Priority"] == "High"


def test_record_validator_valid_and_duplicate():
    mapper = ColumnMapper()
    normalizer = DataNormalizer()
    validator = RecordValidator()

    raw_parsed = [
        {
            "raw_data": {"Project ID": "P001", "Project Name": "Alpha", "Client": "ABC", "Status": "Active"},
            "source": {"file": "projects.xlsx", "sheet": "Sheet1", "row": 2, "file_hash": "abc", "dataset_version": "v1"}
        },
        {
            "raw_data": {"Project ID": "P001", "Project Name": "Alpha Duplicate", "Client": "ABC", "Status": "Active"},
            "source": {"file": "projects.xlsx", "sheet": "Sheet1", "row": 3, "file_hash": "abc", "dataset_version": "v1"}
        },
    ]

    valid, invalid = validator.validate_records(raw_parsed, mapper, normalizer)

    assert len(valid) == 1
    assert valid[0].project_id == "P001"
    assert len(invalid) == 1
    assert "Duplicate project_id" in invalid[0]["errors"][0]


def test_record_validator_missing_required_fields():
    mapper = ColumnMapper()
    normalizer = DataNormalizer()
    validator = RecordValidator(allow_missing_id=False)

    raw_parsed = [
        {
            "raw_data": {"Client": "Unknown Client"},
            "source": {"file": "projects.xlsx", "sheet": "Sheet1", "row": 2, "file_hash": "abc", "dataset_version": "v1"}
        }
    ]

    valid, invalid = validator.validate_records(raw_parsed, mapper, normalizer)

    assert len(valid) == 0
    assert len(invalid) == 1
    assert "Missing both project_id and project_name" in invalid[0]["errors"][0]
