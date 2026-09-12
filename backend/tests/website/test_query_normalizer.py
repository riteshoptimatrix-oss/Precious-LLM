import pytest
from app.website.search.query import WebsiteQueryNormalizer

def test_query_normalizer_basic():
    tokens = WebsiteQueryNormalizer.normalize_query("How to apply for F1 visa?")
    assert "f1" in tokens
    assert "visa" in tokens
    # expanded synonyms for f1
    assert "student visa" in tokens or "f-1" in tokens

def test_query_normalizer_empty():
    assert WebsiteQueryNormalizer.normalize_query("") == []
    assert WebsiteQueryNormalizer.normalize_query(None) == []
