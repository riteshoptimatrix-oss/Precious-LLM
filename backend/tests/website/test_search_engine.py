import pytest
from unittest.mock import AsyncMock, MagicMock
from app.website.search.search_engine import WebsiteSearchEngine

@pytest.mark.asyncio
async def test_search_engine_ranking():
    db = MagicMock()
    engine = WebsiteSearchEngine(db, min_relevance_score=5.0)
    
    mock_chunks = [
        {
            "canonical_url": "https://www.preciousedu.in/visa",
            "title": "Student Visa Guidance",
            "section": "F1 Visa Overview",
            "content": "Detailed instructions on F1 student visa interview preparation."
        },
        {
            "canonical_url": "https://www.preciousedu.in/about",
            "title": "About Precious Education",
            "section": "Company History",
            "content": "Founded in India to help students study abroad."
        }
    ]
    
    engine.repo.get_active_chunks = AsyncMock(return_value=mock_chunks)
    
    results = await engine.search("F1 Student Visa", top_k=1)
    assert len(results) == 1
    assert results[0]["title"] == "Student Visa Guidance"

@pytest.mark.asyncio
async def test_search_engine_empty_query():
    db = MagicMock()
    engine = WebsiteSearchEngine(db)
    results = await engine.search("")
    assert results == []
