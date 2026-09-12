"""
Precious Edu LLM — CORS Middleware Configuration

Configures Cross-Origin Resource Sharing to allow
the PHP frontend to communicate with the FastAPI backend.
"""

from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings


def get_cors_config() -> dict:
    """
    Get CORS middleware configuration.

    Returns a dict of kwargs for CORSMiddleware.
    """
    settings = get_settings()

    return {
        "allow_origins": settings.cors_origins_list,
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Authorization",
            "X-Request-ID",
        ],
        "max_age": 600,  # Cache preflight responses for 10 minutes
    }
