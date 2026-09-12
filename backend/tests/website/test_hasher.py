import pytest
from app.website.processing.hasher import ContentHasher

def test_content_hasher_deterministic():
    text1 = "Precious Education overseas counseling"
    text2 = "  Precious   Education overseas counseling \n"
    
    hash1 = ContentHasher.compute_hash(text1)
    hash2 = ContentHasher.compute_hash(text2)
    
    assert hash1 != ""
    assert hash1 == hash2

def test_content_hasher_empty():
    assert ContentHasher.compute_hash("") == ""
    assert ContentHasher.compute_hash(None) == ""
