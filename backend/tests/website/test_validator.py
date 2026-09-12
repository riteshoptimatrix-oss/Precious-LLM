import pytest
from app.website.processing.validator import WebsiteQualityValidator

def test_validator_empty_dataset():
    is_valid, report = WebsiteQualityValidator.validate_dataset([], [])
    assert is_valid is False
    assert "Dataset contains 0 pages." in report["errors"]
    assert "Dataset contains 0 chunks." in report["errors"]

def test_validator_valid_dataset():
    pages = [
        {"url": "https://www.preciousedu.in/", "title": "Home", "content": "Welcome", "content_hash": "hash1"}
    ]
    chunks = [
        {"canonical_url": "https://www.preciousedu.in/", "content": "Welcome"}
    ]
    is_valid, report = WebsiteQualityValidator.validate_dataset(pages, chunks)
    assert is_valid is True
    assert report["total_pages"] == 1
    assert report["total_chunks"] == 1
    assert len(report["errors"]) == 0
