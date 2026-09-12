"""
Precious AI — Website Local Search Engine
"""

import logging
from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.website.repositories.chunk_repository import WebsiteChunkRepository
from app.website.search.query import WebsiteQueryNormalizer
from app.website.search.scorer import RelevanceScorer

logger = logging.getLogger(__name__)


class WebsiteSearchEngine:
    """
    Local deterministic search engine for website knowledge chunks.
    """

    def __init__(self, db: AsyncIOMotorDatabase, min_relevance_score: float = 5.0):
        self.db = db
        self.repo = WebsiteChunkRepository(db)
        self.min_score = min_relevance_score

    async def search(self, query_text: str, top_k: int = 3, intent: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves top K relevant website chunks for query_text and intent.
        """
        if not query_text or not query_text.strip():
            return []

        tokens = WebsiteQueryNormalizer.normalize_query(query_text)
        if not tokens:
            return []

        active_chunks = await self.repo.get_active_chunks(limit=2000)
        if not active_chunks:
            return []

        scored_results = []
        for chunk in active_chunks:
            score = RelevanceScorer.score_chunk(chunk, query_text, tokens, intent=intent)
            if score >= self.min_score:
                chunk_copy = dict(chunk)
                chunk_copy["score"] = score
                scored_results.append({
                    "score": score,
                    "chunk": chunk_copy
                })

        # Sort by score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate top results
        deduped = []
        seen_sections = set()

        for item in scored_results:
            c = item["chunk"]
            key = f"{c.get('canonical_url')}:{c.get('section')}"
            if key not in seen_sections:
                seen_sections.add(key)
                deduped.append(c)
                if len(deduped) >= top_k:
                    break

        return deduped
