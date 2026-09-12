"""
Precious AI — Website Refresh Service

Incremental refresh: re-crawls the website but only re-indexes pages whose
SHA-256 content hash has changed since the last active version.

Change classification:
  - NEW:       URL not in previous index
  - CHANGED:   URL exists but content_hash differs
  - UNCHANGED: URL exists and content_hash matches
  - REMOVED:   URL was in previous index but not found in current crawl

Supports explicit version rollback via rollback_to_version().
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

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


class WebsiteRefreshService:
    """
    Incremental website knowledge refresh service.
    Only re-chunks and re-indexes pages that have changed since the last crawl.
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

    async def run_refresh(self) -> CrawlRunSummary:
        """
        Executes incremental website refresh.

        Steps:
        1. Load existing active page hashes from MongoDB.
        2. Crawl the live website.
        3. Classify each page as NEW / CHANGED / UNCHANGED.
        4. Only parse + chunk NEW and CHANGED pages.
        5. Build combined page list (changed + unchanged) for the new version.
        6. Validate, persist, register, and activate the new version.

        Returns:
            CrawlRunSummary with per-category page counts.
        """
        logger.info(f"Starting incremental refresh of {self.base_url}")

        # Step 1: Load existing hashes from MongoDB
        existing_hashes: Dict[str, str] = {}  # canonical_url -> content_hash
        try:
            existing_pages = await self.page_repo.get_active_pages()
            for p in existing_pages:
                url = p.get("canonical_url", "")
                h = p.get("content_hash", "")
                if url and h:
                    existing_hashes[url] = h
            logger.info(f"Loaded {len(existing_hashes)} existing page hashes.")
        except Exception as e:
            logger.warning(f"Could not load existing hashes (first run?): {e}")

        # Step 2: Crawl
        fetched_pages, summary = await self.crawler.crawl()
        logger.info(f"Refresh crawl complete: {len(fetched_pages)} pages fetched")

        if not fetched_pages:
            summary.status = "FAILED"
            summary.errors.append({"error": "No pages fetched during refresh crawl."})
            await self.crawl_run_repo.save_run(summary.model_dump())
            return summary

        website_version = summary.website_version
        crawled_urls: Set[str] = set()

        # Step 3: Classify and process
        new_page_docs: List[Dict[str, Any]] = []
        unchanged_page_docs: List[Dict[str, Any]] = []
        chunk_docs: List[Dict[str, Any]] = []

        for page in fetched_pages:
            if not page.raw_html:
                continue

            crawled_urls.add(page.canonical_url)

            try:
                # Parse for hash computation
                parsed = self.parser.parse(page.raw_html, url=page.canonical_url)
                page_content = parsed.get("clean_content", "")
                page_hash = ContentHasher.compute_hash(page_content)

                existing_hash = existing_hashes.get(page.canonical_url)

                if existing_hash is None:
                    # NEW page
                    status = "new"
                    summary.pages_new += 1
                elif existing_hash != page_hash:
                    # CHANGED page
                    status = "changed"
                    summary.pages_changed += 1
                else:
                    # UNCHANGED page — skip re-chunking
                    status = "unchanged"
                    summary.pages_unchanged += 1

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
                    "change_status": status,
                    "active": True,
                    "indexed_at": datetime.now(timezone.utc).isoformat(),
                }

                if status in ("new", "changed"):
                    new_page_docs.append(page_doc)
                    chunks = self.chunker.create_chunks(parsed, website_version)
                    chunk_docs.extend(chunks)
                else:
                    unchanged_page_docs.append(page_doc)

            except Exception as e:
                logger.warning(f"Refresh extraction error for {page.canonical_url}: {e}")
                summary.pages_failed += 1
                summary.errors.append({"url": page.canonical_url, "error": str(e)})

        # Step 4: Detect removed pages
        removed_urls = set(existing_hashes.keys()) - crawled_urls
        summary.pages_removed = len(removed_urls)
        if removed_urls:
            logger.info(f"{len(removed_urls)} pages removed since last crawl.")

        all_page_docs = new_page_docs + unchanged_page_docs
        summary.chunks_created = len(chunk_docs)

        # Step 5: Validate
        is_valid, validation_report = self.validator.validate_dataset(all_page_docs, chunk_docs)
        if not is_valid:
            logger.error(f"Refresh validation failed: {validation_report['errors']}")
            summary.status = "VALIDATION_FAILED"
            summary.errors.append({"validation": validation_report})
            await self.crawl_run_repo.save_run(summary.model_dump())
            return summary

        # Step 6: Persist, register, activate
        await self.page_repo.save_pages(all_page_docs)

        # Only persist chunks for new/changed pages
        if chunk_docs:
            await self.chunk_repo.save_chunks(chunk_docs)

        await self.version_manager.register_version(
            website_version=website_version,
            crawl_id=summary.crawl_id,
            pages_count=len(all_page_docs),
            chunks_count=len(chunk_docs),
            extra_meta={
                "pages_new": summary.pages_new,
                "pages_changed": summary.pages_changed,
                "pages_unchanged": summary.pages_unchanged,
                "pages_removed": summary.pages_removed,
                "is_refresh": True,
            },
        )
        await self.version_manager.activate_version(website_version)

        summary.status = "COMPLETED"
        summary.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        await self.crawl_run_repo.save_run(summary.model_dump())

        logger.info(
            f"Refresh complete — new: {summary.pages_new}, changed: {summary.pages_changed}, "
            f"unchanged: {summary.pages_unchanged}, removed: {summary.pages_removed}"
        )
        return summary

    async def rollback_to_version(self, website_version: str) -> Dict[str, Any]:
        """
        Rolls back the active website knowledge dataset to a previous version.

        Args:
            website_version: Target version ID to rollback to.

        Returns:
            Rollback summary dict from WebsiteVersionManager.
        """
        logger.warning(f"Initiating rollback to version: {website_version}")
        return await self.version_manager.rollback_to_version(website_version)
