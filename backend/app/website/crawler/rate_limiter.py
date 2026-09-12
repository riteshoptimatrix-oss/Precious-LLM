"""
Precious AI — Crawler Rate Limiter & Concurrency Throttler
"""

import asyncio
import time
from typing import Optional


class RateLimiter:
    """
    Controls request concurrency and delay spacing between HTTP requests.
    """

    def __init__(self, delay_seconds: float = 0.5, max_concurrency: int = 2):
        self.delay_seconds = delay_seconds
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Acquires semaphore slot and enforces minimum delay spacing between requests.
        """
        await self.semaphore.acquire()
        async with self._lock:
            now = time.perf_counter()
            elapsed = now - self.last_request_time
            if elapsed < self.delay_seconds:
                await asyncio.sleep(self.delay_seconds - elapsed)
            self.last_request_time = time.perf_counter()

    def release(self) -> None:
        """
        Releases semaphore slot.
        """
        self.semaphore.release()
