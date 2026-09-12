"""
Precious Edu LLM — Central API Router

Aggregates all API route modules under /api prefix.
"""

from fastapi import APIRouter
from app.api.routes import health, sessions, chat

api_router = APIRouter(prefix="/api")

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(sessions.router, tags=["Sessions"])
api_router.include_router(chat.router, tags=["Chat"])
