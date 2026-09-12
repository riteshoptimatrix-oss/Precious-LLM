"""
Unit Tests — RateLimiter

Covers: concurrency limits and delay enforcement.
"""

import pytest
import asyncio
import time
from app.website.crawler.rate_limiter import RateLimiter

@pytest.mark.asyncio
class TestRateLimiter:
    async def test_enforces_delay(self):
        limiter = RateLimiter(delay_seconds=0.2, max_concurrency=1)
        
        start = time.perf_counter()
        await limiter.acquire()
        limiter.release()
        
        await limiter.acquire()
        limiter.release()
        end = time.perf_counter()
        
        # Second acquire should have waited at least 0.2s
        assert (end - start) >= 0.2

    async def test_concurrency_limit(self):
        limiter = RateLimiter(delay_seconds=0.1, max_concurrency=2)
        
        # Acquire 2 slots immediately
        await limiter.acquire()
        await limiter.acquire()
        
        # Third acquire should block until a release
        task = asyncio.create_task(limiter.acquire())
        
        # Allow event loop to run slightly
        await asyncio.sleep(0.05)
        assert not task.done()
        
        limiter.release()
        await asyncio.sleep(0.05)
        
        # Now it should be done, but it might wait for the delay
        await task
        assert task.done()
        limiter.release()
        limiter.release()
