"""
Precious Edu LLM — Rate Limiter Middleware

Simple in-memory rate limiter using a token bucket algorithm.
Limits requests per IP address to prevent abuse.

NOTE: For production, consider using Redis-backed rate limiting
for multi-process/multi-server deployments.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.config import get_settings

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    In-memory rate limiter middleware.

    Uses a simple sliding window counter per IP address.
    Configurable via RATE_LIMIT_PER_MINUTE environment variable.
    """

    def __init__(self, app):
        super().__init__(app)
        settings = get_settings()
        self.rate_limit = settings.RATE_LIMIT_PER_MINUTE
        self.window_seconds = 60
        # Dict[ip_address] -> List of request timestamps
        self._requests: Dict[str, list] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, considering proxy headers."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, client_ip: str) -> Tuple[bool, int]:
        """
        Check if a client IP has exceeded the rate limit.

        Returns:
            Tuple of (is_limited, remaining_requests)
        """
        now = time.time()
        window_start = now - self.window_seconds

        # Remove expired timestamps
        self._requests[client_ip] = [
            ts for ts in self._requests[client_ip] if ts > window_start
        ]

        request_count = len(self._requests[client_ip])

        if request_count >= self.rate_limit:
            return True, 0

        return False, self.rate_limit - request_count

    async def dispatch(self, request: Request, call_next):
        """Process request through rate limiter."""
        # Skip rate limiting for health checks
        if request.url.path == "/api/health":
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        is_limited, remaining = self._is_rate_limited(client_ip)

        if is_limited:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": (
                            f"Too many requests. Limit is {self.rate_limit} "
                            f"requests per minute."
                        ),
                    }
                },
            )

        # Record this request
        self._requests[client_ip].append(time.time())

        # Process the request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.rate_limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining - 1)

        return response
