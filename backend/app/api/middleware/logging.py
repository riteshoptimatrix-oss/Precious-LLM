"""
Precious Edu LLM — Request Logging Middleware

Logs incoming requests and outgoing responses with timing information.
Sensitive data (message content, auth tokens) is NOT logged.
"""

import time
import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

logger = logging.getLogger("app.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs request/response metadata.

    Logs:
    - Request method, path, client IP
    - Response status code
    - Request duration in milliseconds
    - Unique request ID for tracing

    Does NOT log:
    - Request body (may contain sensitive user messages)
    - Response body
    - Authorization headers
    """

    async def dispatch(self, request: Request, call_next):
        """Log request metadata and timing."""
        # Generate a unique request ID for tracing
        request_id = str(uuid.uuid4())[:8]

        # Extract request metadata
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        # Start timing
        start_time = time.time()

        # Log incoming request
        logger.info(
            f"[{request_id}] --> {method} {path} from {client_ip}"
        )

        try:
            # Process the request
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log response
            logger.info(
                f"[{request_id}] <-- {response.status_code} "
                f"({duration_ms:.1f}ms)"
            )

            # Add request ID header for client-side tracing
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] !!! Error after {duration_ms:.1f}ms: {e}"
            )
            raise
