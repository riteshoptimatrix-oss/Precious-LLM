import pytest
from app.website.processing.chunker import WebsiteChunker

def test_chunker_single_short_section():
    chunker = WebsiteChunker(max_words_per_chunk=100)
    parsed_page = {
        "url": "https://www.preciousedu.in/about",
        "title": "About Us",
        "page_type": "about",
        "sections": [
            {
                "heading": "Overview",
                "content": "Precious Education is a leading abroad study consultancy in India."
            }
        ]
    }
    chunks = chunker.create_chunks(parsed_page, "v1.0.0")

    assert len(chunks) == 1
    assert chunks[0]["canonical_url"] == "https://www.preciousedu.in/about"
    assert chunks[0]["title"] == "About Us"
    assert chunks[0]["section"] == "Overview"
    assert chunks[0]["website_version"] == "v1.0.0"
    assert chunks[0]["active"] is True

def test_chunker_long_section_sliding_window():
    chunker = WebsiteChunker(max_words_per_chunk=10, overlap_words=2)
    words = [f"word{i}" for i in range(25)]
    parsed_page = {
        "url": "https://www.preciousedu.in/long-page",
        "title": "Long Page",
        "page_type": "other",
        "sections": [
            {
                "heading": "Long Section",
                "content": " ".join(words)
            }
        ]
    }
    chunks = chunker.create_chunks(parsed_page, "v1.0.0")

    assert len(chunks) > 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
