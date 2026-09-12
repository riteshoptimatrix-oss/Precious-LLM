"""
Precious Edu LLM — FastAPI Dependency Injection Providers

Provides shared dependencies for route handlers:
- Database access
- Settings
- Service instances
- Model/inference engine (when available)
"""

from fastapi import Depends

from app.config import Settings, get_settings
from app.db.mongodb import get_database


async def get_db():
    """
    Dependency: Get the MongoDB database instance.

    Yields the database object for use in route handlers.
    """
    db = await get_database()
    return db


def get_app_settings() -> Settings:
    """
    Dependency: Get application settings.

    Returns the cached Settings instance.
    """
    return get_settings()


def get_llm():
    """
    Dependency: Get the global LLMService instance.
    """
    from app.llm import get_llm_service
    return get_llm_service()


async def get_chat_service(db=Depends(get_db)):
    """
    Dependency: Get the ChatService instance initialized with LLMService and MongoDB.
    """
    from app.services.chat_service import ChatService
    from app.llm import get_llm_service
    llm = get_llm_service()
    return ChatService(db=db, generator=llm)
