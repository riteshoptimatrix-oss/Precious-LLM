"""
Precious Edu LLM — Centralized Logging Setup

Configures structured logging for the application.
Ensures sensitive details are not leaked into public log streams.
"""

import logging
import sys
from app.core.config import get_settings


def setup_logging() -> None:
    """
    Setup logging configuration according to APP LOG_LEVEL setting.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Silence verbose third-party loggers if necessary
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
