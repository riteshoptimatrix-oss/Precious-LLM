"""
Precious Edu LLM — Global Error Handler

Catches unhandled exceptions and returns structured JSON error responses.
Prevents raw stack traces from reaching the client.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Handle FastAPI/Starlette HTTP exceptions.

    Converts them to our standard error response format.
    """
    # If detail is already a dict with our format, use it directly
    if isinstance(exc.detail, dict) and "code" in exc.detail:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )

    # Otherwise, wrap in our standard format
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": _status_to_code(exc.status_code),
                "message": str(exc.detail),
            }
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    """
    Handle Pydantic validation errors.

    Formats validation errors into a user-friendly structure.
    """
    errors = exc.errors()
    first_error = errors[0] if errors else {}

    # Extract field name from the location path
    field = " -> ".join(str(loc) for loc in first_error.get("loc", []))
    message = first_error.get("msg", "Validation error")

    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": f"{field}: {message}" if field else message,
                "details": {
                    "errors": [
                        {
                            "field": " -> ".join(str(loc) for loc in e.get("loc", [])),
                            "message": e.get("msg", ""),
                            "type": e.get("type", ""),
                        }
                        for e in errors
                    ]
                },
            }
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions.

    Logs the full error but returns a generic message to the client
    to prevent information leakage.
    """
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Please try again later.",
            }
        },
    )


def register_error_handlers(app: FastAPI):
    """Register all error handlers with the FastAPI application."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)


def _status_to_code(status_code: int) -> str:
    """Map HTTP status codes to error codes."""
    mapping = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMIT_EXCEEDED",
        500: "INTERNAL_ERROR",
        501: "NOT_IMPLEMENTED",
        503: "SERVICE_UNAVAILABLE",
    }
    return mapping.get(status_code, "UNKNOWN_ERROR")
