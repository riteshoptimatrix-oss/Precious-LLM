"""
Precious AI — Website Knowledge Service

Search facade for the chatbot pipeline:
  - Wraps WebsiteSearchEngine with context formatting
  - Formats retrieved chunks into <website_knowledge> context block
  - Provides health status reporting (version, chunk count, last crawl)
"""

import logging
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.website.search.search_engine import WebsiteSearchEngine
from app.website.repositories.version_repository import WebsiteVersionRepository
from app.website.repositories.chunk_repository import WebsiteChunkRepository
from app.website.repositories.crawl_run_repository import CrawlRunRepository

logger = logging.getLogger(__name__)


class WebsiteKnowledgeService:
    """
    Search facade for the website knowledge engine.
    Used by KnowledgeEngine and ConversationEngine at chat inference time.
    No crawling occurs here — all reads are against the active MongoDB index.
    """

    def __init__(self, db: AsyncIOMotorDatabase, top_k: int = 3, min_score: float = 5.0):
        self.db = db
        self.top_k = top_k
        self.search_engine = WebsiteSearchEngine(db, min_relevance_score=min_score)
        self.version_repo = WebsiteVersionRepository(db)
        self.chunk_repo = WebsiteChunkRepository(db)
        self.crawl_run_repo = CrawlRunRepository(db)

    def validate_chunk_relevance(self, chunk: Dict[str, Any], intent: Optional[str]) -> bool:
        """
        Knowledge Match Validation: Ensures selected chunk aligns with predicted intent.
        Rejects legal/policy chunks for general queries.
        """
        url = chunk.get("canonical_url", "").lower()
        title = chunk.get("title", "").lower()
        section = chunk.get("section", "").lower()
        content = chunk.get("content", "").lower()

        is_legal_page = ("privacy" in url or "terms" in url or "privacy" in title or "terms" in title)

        if is_legal_page and intent not in ["privacy_policy", "terms_conditions"]:
            logger.warning(f"[RELEVANCE REJECTED] Legal page '{title}' rejected for intent '{intent}'")
            return False

        if intent == "ielts_classes":
            has_ielts_context = any(w in content or w in section for w in ["ielts", "coaching", "faculty", "education", "consultant"])
            if not has_ielts_context:
                logger.warning(f"[RELEVANCE REJECTED] Chunk '{section}' missing IELTS context for intent '{intent}'")
                return False

        if intent == "services":
            has_services_context = any(w in content or w in section for w in ["service", "counseling", "counselling", "education", "consultant", "visa"])
            if not has_services_context:
                logger.warning(f"[RELEVANCE REJECTED] Chunk '{section}' missing Services context for intent '{intent}'")
                return False

        if intent == "visa_services":
            has_visa_context = any(w in content or w in section for w in ["visa", "study", "abroad", "education", "consultant", "university", "college", "counseling", "australia", "canada", "uk", "usa"])
            if not has_visa_context:
                logger.warning(f"[RELEVANCE REJECTED] Chunk '{section}' missing Visa context for intent '{intent}'")
                return False

        return True

    async def search(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the top-K most relevant active website chunks for a query and intent.
        """
        k = top_k or self.top_k
        raw_chunks = await self.search_engine.search(query_text, top_k=k, intent=intent)
        valid_chunks = [c for c in raw_chunks if self.validate_chunk_relevance(c, intent)]
        return valid_chunks

    async def get_context_block(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        intent: Optional[str] = None,
    ) -> Optional[str]:
        """
        Retrieves website chunks for a query and formats them into a
        <website_knowledge> XML-style context string for LLM prompt injection.
        """
        chunks = await self.search(query_text, top_k=top_k, intent=intent)
        if not chunks:
            return None

        lines: List[str] = [
            "The following information is from the official Precious Education website "
            "(https://www.preciousedu.in/):\n"
        ]

        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("title", "Precious Education")
            section = chunk.get("section", "")
            content = chunk.get("content", "").strip()
            url = chunk.get("canonical_url", "https://www.preciousedu.in/")

            heading = f"{title} — {section}" if section and section != title else title
            lines.append(f"[Source {i}] {heading} ({url})")
            lines.append(content)
            lines.append("")

        return "\n".join(lines).strip()

    def get_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extracts source reference dicts from chunk results for ChatResponse.sources field.

        Returns:
            List of {"url": ..., "title": ..., "section": ...} dicts.
        """
        sources = []
        seen = set()
        for chunk in chunks:
            url = chunk.get("canonical_url", "")
            key = f"{url}:{chunk.get('section', '')}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "url": url,
                    "title": chunk.get("title", "Precious Education"),
                    "section": chunk.get("section", ""),
                })
        return sources

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Returns health metadata for the /api/health/website endpoint.

        Returns:
            Dict with: is_ready, active_version, pages_count, chunks_count,
                       last_crawl_at, last_crawl_status
        """
        try:
            active_version = await self.version_repo.get_active_version()
            active_chunks = await self.chunk_repo.get_active_chunks(limit=10000)
            latest_run = await self.crawl_run_repo.get_latest_run()

            is_ready = bool(active_version and len(active_chunks) > 0)

            return {
                "is_ready": is_ready,
                "active_version": active_version.get("website_version") if active_version else None,
                "pages_count": active_version.get("pages_count", 0) if active_version else 0,
                "chunks_count": len(active_chunks),
                "activated_at": active_version.get("activated_at") if active_version else None,
                "last_crawl_at": latest_run.get("completed_at") if latest_run else None,
                "last_crawl_status": latest_run.get("status") if latest_run else None,
                "last_crawl_id": latest_run.get("crawl_id") if latest_run else None,
            }
        except Exception as e:
            logger.error(f"Website health status error: {e}")
            return {
                "is_ready": False,
                "error": str(e),
            }
