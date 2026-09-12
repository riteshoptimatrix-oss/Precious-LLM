"""
Precious Edu LLM — FastAPI Application Factory

Creates and configures the FastAPI application with:
- Lifespan events (startup/shutdown for DB connection & indexes)
- Middleware (CORS, logging, error handling)
- Central route registration via api_router
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.api.router import api_router
from app.api.middleware.error_handler import register_error_handlers
from app.api.middleware.logging import RequestLoggingMiddleware
from app.db.mongodb import connect_to_mongodb, close_mongodb_connection, get_database
from app.db.indexes import create_database_indexes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager handling startup and shutdown events.
    """
    settings = get_settings()

    # --- Startup ---
    setup_logging()
    logger.info(f"Starting {settings.APP_NAME} backend...")
    logger.info(f"Environment: {settings.APP_ENV}")

    # Connect to MongoDB
    await connect_to_mongodb()

    # Create collection indexes
    db = await get_database()
    await create_database_indexes(db)

    # Initialize Custom LLM Service (Loads model ONCE into memory)
    try:
        from app.llm import get_llm_service
        llm_service = get_llm_service()
        llm_service.initialize()
        logger.info("Custom LLM Service initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Custom LLM Service: {e}")

    logger.info("Application startup complete.")

    yield

    # --- Shutdown ---
    logger.info(f"Shutting down {settings.APP_NAME} backend...")
    await close_mongodb_connection()
    logger.info("Application shutdown complete.")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "A domain-specific conversational AI chatbot backend powered by "
            "custom-trained decoder-only Transformer language model."
        ),
        version="0.1.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?" if settings.is_development else None,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)

    # Register global exception handlers
    register_error_handlers(app)

    # Include central API router
    app.include_router(api_router)

    return app


# Application entry point instance
app = create_app()
