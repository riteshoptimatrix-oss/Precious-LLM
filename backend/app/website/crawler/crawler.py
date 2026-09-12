"""
Precious AI — Main Web Crawler Engine

Coordinates URL discovery, queue management, depth bounds, page bounds,
robots compliance, rate limiting, and HTTP fetching.
"""

import asyncio
import time
import logging
from typing import List, Dict, Set, Optional, Tuple

from app.website.crawler.models import CrawlStatus, DiscoveredURL, FetchedPage, CrawlRunSummary
from app.website.crawler.url_normalizer import URLNormalizer
from app.website.crawler.robots import RobotsChecker
from app.website.crawler.rate_limiter import RateLimiter
from app.website.crawler.fetcher import HTTPFetcher
from app.website.crawler.url_discovery import URLDiscoverer

logger = logging.getLogger(__name__)


class WebCrawler:
    """
    Production Web Crawler for Precious Education website.
    """

    def __init__(
        self,
        base_url: str = "https://www.preciousedu.in/",
        max_depth: int = 5,
        max_pages: int = 500,
        delay_seconds: float = 0.5,
        concurrency: int = 2,
        user_agent: str = "PreciousAICrawler/1.0"
    ):
        self.base_url = base_url
        self.max_depth = max_depth
        self.max_pages = max_pages

        self.normalizer = URLNormalizer()
        self.robots = RobotsChecker(user_agent=user_agent)
        self.rate_limiter = RateLimiter(delay_seconds=delay_seconds, max_concurrency=concurrency)
        self.fetcher = HTTPFetcher(rate_limiter=self.rate_limiter, user_agent=user_agent)
        self.discoverer = URLDiscoverer(normalizer=self.normalizer)

    async def crawl(self) -> Tuple[List[FetchedPage], CrawlRunSummary]:
        """
        Executes web crawl starting from base_url.
        """
        crawl_id = f"crawl-{int(time.time())}"
        version_id = f"web-v{int(time.time())}"
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        summary = CrawlRunSummary(
            crawl_id=crawl_id,
            website_version=version_id,
            base_url=self.base_url,
            started_at=started_at
        )

        norm_start = self.normalizer.normalize(self.base_url)
        if not norm_start:
            summary.status = "FAILED"
            summary.errors.append({"error": f"Invalid start URL: {self.base_url}"})
            return [], summary

        # 1. Inspect robots.txt
        await self.robots.fetch_and_parse(norm_start)

        # 2. Initialize Queue & Visited Set
        queue: List[DiscoveredURL] = [DiscoveredURL(url=norm_start, normalized_url=norm_start, depth=0)]
        visited: Set[str] = {norm_start}
        fetched_pages: List[FetchedPage] = []

        # 3. Discover from Sitemap
        sitemap_urls = await self.discoverer.discover_from_sitemap(norm_start)
        for s_url in sitemap_urls:
            if s_url not in visited and len(queue) < self.max_pages:
                visited.add(s_url)
                queue.append(DiscoveredURL(url=s_url, normalized_url=s_url, depth=1, discovered_from="sitemap.xml"))

        summary.pages_discovered = len(visited)

        # 4. Crawl Loop
        while queue and len(fetched_pages) < self.max_pages:
            current_item = queue.pop(0)
            target_url = current_item.normalized_url

            # Check robots.txt permission
            if not self.robots.is_allowed(target_url):
                summary.pages_failed += 1
                summary.errors.append({"url": target_url, "reason": "Blocked by robots.txt"})
                continue

            # Fetch page
            page = await self.fetcher.fetch(
                url=target_url,
                normalized_url=target_url,
                depth=current_item.depth
            )
            summary.pages_fetched += 1

            if page.error:
                summary.pages_failed += 1
                summary.errors.append({"url": target_url, "error": page.error})
            else:
                fetched_pages.append(page)

                # Discover new links if depth < max_depth
                if current_item.depth < self.max_depth:
                    new_links = self.discoverer.discover_from_html(page.raw_html, base_url=target_url)
                    for link in new_links:
                        if link not in visited and (len(visited) + len(queue)) < self.max_pages:
                            visited.add(link)
                            queue.append(DiscoveredURL(
                                url=link,
                                normalized_url=link,
                                depth=current_item.depth + 1,
                                discovered_from=target_url
                            ))
                            summary.pages_discovered += 1

        summary.completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        summary.status = "COMPLETED"
        return fetched_pages, summary
