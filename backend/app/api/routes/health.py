"""
Precious Edu LLM — Health Check Routes

GET /api/health — Overall application & database health
GET /api/health/db — Dedicated MongoDB connectivity check
"""

import logging
from typing import Dict
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_db, get_app_settings
from app.schemas.common import HealthResponse, DatabaseHealthResponse
from app.core.config import Settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application Health Check",
    description="Returns the health status of application and MongoDB.",
)
async def health_check(
    db=Depends(get_db),
    settings: Settings = Depends(get_app_settings)
):
    """
    Check application and database status.
    """
    health_status = "ok"
    db_status = "connected"

    try:
        await db.command("ping")
    except Exception as e:
        logger.error(f"MongoDB health ping failed: {e}")
        health_status = "degraded"
        db_status = "disconnected"

    return HealthResponse(
        status=health_status,
        service=settings.APP_NAME,
        database=db_status,
        version="0.1.0"
    )


@router.get(
    "/health/db",
    response_model=DatabaseHealthResponse,
    summary="Database Health Check",
    description="Checks dedicated MongoDB connection status.",
)
async def db_health_check(
    db=Depends(get_db),
    settings: Settings = Depends(get_app_settings)
):
    """
    Dedicated MongoDB health endpoint.
    """
    try:
        await db.command("ping")
        return DatabaseHealthResponse(
            status="ok",
            database="connected",
            database_name=settings.MONGODB_DB_NAME
        )
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database connection failed: {str(e)}"
        )


@router.get(
    "/health/llm",
    summary="LLM Model Health & Readiness Check",
    description="Checks readiness, version, parameter count, device, and checkpoint status of Custom LLM."
)
async def llm_health_check():
    """
    Dedicated Custom LLM health & readiness endpoint.
    """
    from app.llm import get_llm_service
    llm = get_llm_service()
    info = llm.get_info()

    if not info.get("is_ready", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Custom LLM model is not loaded into memory."
        )
    return info


@router.get(
    "/health/website",
    summary="Website Knowledge Engine Health Check",
    description="Reports active version, chunk count, page count, and last crawl status for the website knowledge index.",
)
async def website_health_check(db=Depends(get_db)):
    """
    Dedicated website knowledge engine health & readiness endpoint.

    Returns:
        is_ready: True if an active version with indexed chunks exists.
        active_version: Current active version ID.
        pages_count: Number of pages in active version.
        chunks_count: Number of active chunks available for search.
        activated_at: ISO timestamp when version was activated.
        last_crawl_at: ISO timestamp of last completed crawl.
        last_crawl_status: Status string of last crawl run.
    """
    try:
        from app.website.services.website_knowledge_service import WebsiteKnowledgeService
        website_service = WebsiteKnowledgeService(db)
        health = await website_service.get_health_status()
        return health
    except Exception as e:
        logger.error(f"Website health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Website knowledge engine health check failed: {str(e)}"
        )


@router.get(
    "/health/structured-qa",
    summary="Structured Q&A Health Check",
    description="Reports collection readiness, active record count, and unique intent count.",
)
async def structured_qa_health_check(db=Depends(get_db)):
    """
    Dedicated structured Q&A knowledge engine health & readiness endpoint.
    """
    try:
        from app.knowledge.structured_repository import StructuredQARepository
        repo = StructuredQARepository(db)
        health = await repo.get_health_status()
        if not health.get("is_ready", False):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Structured Q&A collection is empty or not initialized."
            )
        return health
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Structured Q&A health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Structured Q&A health check failed: {str(e)}"
        )

