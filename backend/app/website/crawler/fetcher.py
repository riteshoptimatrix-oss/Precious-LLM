"""
Precious AI — Local HTTP Fetcher

Executes HTTP requests with rate limiting, timeouts, retries, content-type checks,
and max size protection.
"""

import asyncio
import time
import logging
from typing import Optional, Tuple
import httpx

from app.website.crawler.models import FetchedPage
from app.website.crawler.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class HTTPFetcher:
    """
    Robust HTTP Fetcher.
    """

    ALLOWED_CONTENT_TYPES = {"text/html", "application/xhtml+xml"}
    MAX_RESPONSE_SIZE = 5 * 1024 * 1024  # 5MB

    def __init__(
        self,
        timeout_sec: float = 15.0,
        max_retries: int = 3,
        user_agent: str = "PreciousAICrawler/1.0",
        rate_limiter: Optional[RateLimiter] = None
    ):
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.user_agent = user_agent
        self.rate_limiter = rate_limiter or RateLimiter()

    async def fetch(self, url: str, normalized_url: str, depth: int = 0) -> FetchedPage:
        """
        Fetches single URL with retries, rate-limiting, and error handling.
        """
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            await self.rate_limiter.acquire()
            t0 = time.perf_counter()
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout_sec,
                    follow_redirects=True,
                    headers=headers
                ) as client:
                    resp = await client.get(url)
                    lat = time.perf_counter() - t0

                    content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()

                    # Content-type check
                    if content_type and not any(ct in content_type for ct in self.ALLOWED_CONTENT_TYPES):
                        return FetchedPage(
                            url=url,
                            normalized_url=normalized_url,
                            canonical_url=str(resp.url),
                            status_code=resp.status_code,
                            content_type=content_type,
                            raw_html="",
                            content_length=0,
                            fetch_duration_sec=lat,
                            crawl_depth=depth,
                            error=f"Unsupported content type: {content_type}"
                        )

                    # Size check
                    raw_text = resp.text
                    if len(raw_text.encode("utf-8")) > self.MAX_RESPONSE_SIZE:
                        return FetchedPage(
                            url=url,
                            normalized_url=normalized_url,
                            canonical_url=str(resp.url),
                            status_code=resp.status_code,
                            content_type=content_type,
                            raw_html="",
                            content_length=len(raw_text),
                            fetch_duration_sec=lat,
                            crawl_depth=depth,
                            error="Response size exceeds 5MB limit"
                        )

                    return FetchedPage(
                        url=url,
                        normalized_url=normalized_url,
                        canonical_url=str(resp.url),
                        status_code=resp.status_code,
                        content_type=content_type,
                        raw_html=raw_text,
                        content_length=len(raw_text),
                        fetch_duration_sec=lat,
                        crawl_depth=depth,
                        error=None
                    )
            except Exception as e:
                last_error = str(e)
                logger.warning(f"HTTP fetch attempt {attempt}/{self.max_retries} failed for {url}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(attempt * 1.0)
            finally:
                self.rate_limiter.release()

        return FetchedPage(
            url=url,
            normalized_url=normalized_url,
            canonical_url=url,
            status_code=0,
            content_type="",
            raw_html="",
            content_length=0,
            fetch_duration_sec=0.0,
            crawl_depth=depth,
            error=f"Fetch failed after {self.max_retries} retries: {last_error}"
        )
