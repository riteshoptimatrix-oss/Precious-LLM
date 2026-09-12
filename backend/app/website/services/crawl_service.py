"""
Precious AI — Crawl Service

Full-crawl orchestrator. Coordinates:
  WebCrawler → HTMLDocumentParser → WebsiteChunker → ContentHasher →
  WebsitePageRepository → WebsiteChunkRepository → WebsiteVersionManager → CrawlRunRepository

Supports dry_run mode for URL discovery without writing to MongoDB.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.website.crawler.crawler import WebCrawler
from app.website.crawler.models import CrawlRunSummary
from app.website.extraction.html_parser import HTMLDocumentParser
from app.website.processing.chunker import WebsiteChunker
from app.website.processing.hasher import ContentHasher
from app.website.processing.validator import WebsiteQualityValidator
from app.website.processing.versioning import WebsiteVersionManager
from app.website.repositories.page_repository import WebsitePageRepository
from app.website.repositories.chunk_repository import WebsiteChunkRepository
from app.website.repositories.crawl_run_repository import CrawlRunRepository

logger = logging.getLogger(__name__)


class CrawlService:
    """
    Orchestrates a full website crawl, extraction, processing, and indexing pipeline.
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        base_url: str = "https://www.preciousedu.in/",
        max_depth: int = 5,
        max_pages: int = 500,
        delay_seconds: float = 0.5,
        concurrency: int = 2,
        user_agent: str = "PreciousAICrawler/1.0",
    ):
        self.db = db
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay_seconds = delay_seconds
        self.concurrency = concurrency
        self.user_agent = user_agent

        self.crawler = WebCrawler(
            base_url=base_url,
            max_depth=max_depth,
            max_pages=max_pages,
            delay_seconds=delay_seconds,
            concurrency=concurrency,
            user_agent=user_agent,
        )
        self.parser = HTMLDocumentParser()
        self.chunker = WebsiteChunker()
        self.validator = WebsiteQualityValidator()
        self.version_manager = WebsiteVersionManager(db)

        self.page_repo = WebsitePageRepository(db)
        self.chunk_repo = WebsiteChunkRepository(db)
        self.crawl_run_repo = CrawlRunRepository(db)

    async def run_crawl(self, dry_run: bool = False) -> CrawlRunSummary:
        """
        Executes a full website crawl.

        Args:
            dry_run: If True, discovers URLs and fetches pages but skips all MongoDB writes
                     and version activation. Useful for pre-flight verification.

        Returns:
            CrawlRunSummary with statistics for the completed run.
        """
        logger.info(
            f"{'[DRY-RUN] ' if dry_run else ''}Starting crawl of {self.base_url}"
        )

        # Phase 1: Crawl
        fetched_pages, summary = await self.crawler.crawl()
        logger.info(
            f"Crawl phase complete: {len(fetched_pages)} pages fetched, "
            f"{summary.pages_failed} failed"
        )

        if dry_run:
            summary.status = "DRY_RUN_COMPLETE"
            logger.info("[DRY-RUN] Skipping extraction, indexing, and version activation.")
            return summary

        if not fetched_pages:
            summary.status = "FAILED"
            summary.errors.append({"error": "No pages were fetched successfully."})
            await self.crawl_run_repo.save_run(summary.model_dump())
            return summary

        # Phase 2: Extract & Parse
        website_version = summary.website_version
        page_docs: List[Dict[str, Any]] = []
        chunk_docs: List[Dict[str, Any]] = []

        for page in fetched_pages:
            if not page.raw_html:
                continue
            try:
                parsed = self.parser.parse(page.raw_html, url=page.canonical_url)

                # Build page document
                page_content = parsed.get("clean_content", "")
                page_hash = ContentHasher.compute_hash(page_content)

                page_doc: Dict[str, Any] = {
                    "canonical_url": page.canonical_url,
                    "url": page.url,
                    "title": parsed.get("title", ""),
                    "headings": parsed.get("headings", []),
                    "page_type": parsed.get("page_type", "other"),
                    "content": page_content,
                    "content_hash": page_hash,
                    "website_version": website_version,
                    "crawl_depth": page.crawl_depth,
                    "status_code": page.status_code,
                    "fetch_duration_sec": page.fetch_duration_sec,
                    "fetched_at": page.fetched_at,
                    "active": True,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }
                page_docs.append(page_doc)

                # Build chunk documents
                chunks = self.chunker.create_chunks(parsed, website_version)
                chunk_docs.extend(chunks)

            except Exception as e:
                logger.warning(f"Extraction error for {page.canonical_url}: {e}")
                summary.pages_failed += 1
                summary.errors.append({"url": page.canonical_url, "error": str(e)})

        summary.pages_new = len(page_docs)
        summary.chunks_created = len(chunk_docs)

        # Phase 3: Validate
        is_valid, validation_report = self.validator.validate_dataset(page_docs, chunk_docs)
        if not is_valid:
            logger.error(f"Dataset validation failed: {validation_report['errors']}")
            summary.status = "VALIDATION_FAILED"
            summary.errors.append({"validation": validation_report})
            await self.crawl_run_repo.save_run(summary.model_dump())
            return summary

        logger.info(
            f"Validation passed: {len(page_docs)} pages, {len(chunk_docs)} chunks"
        )

        # Phase 4: Persist
        await self.page_repo.save_pages(page_docs)
        await self.chunk_repo.save_chunks(chunk_docs)
        logger.info("Pages and chunks persisted to MongoDB.")

        # Phase 5: Register & Activate Version
        await self.version_manager.register_version(
            website_version=website_version,
            crawl_id=summary.crawl_id,
            pages_count=len(page_docs),
            chunks_count=len(chunk_docs),
        )
        await self.version_manager.activate_version(website_version)
        logger.info(f"Version '{website_version}' activated.")

        # Phase 6: Save Crawl Run Log
        summary.status = "COMPLETED"
        summary.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        await self.crawl_run_repo.save_run(summary.model_dump())

        logger.info(
            f"Crawl complete — version: {website_version}, "
            f"pages: {len(page_docs)}, chunks: {len(chunk_docs)}"
        )
        return summary
