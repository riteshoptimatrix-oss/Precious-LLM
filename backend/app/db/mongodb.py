"""
Precious Edu LLM — MongoDB Client

Manages the MongoDB connection lifecycle:
- Async connection via Motor (async MongoDB driver)
- Connection pooling
- Database singleton access
- Startup/shutdown hooks

The connection is established during FastAPI lifespan startup
and closed during shutdown.
"""

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level client and database references
_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def connect_to_mongodb() -> None:
    """
    Establish connection to MongoDB.

    Called during FastAPI lifespan startup.
    Creates an async Motor client with connection pooling.
    """
    global _client, _database
    settings = get_settings()

    logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}...")

    _client = AsyncIOMotorClient(
        settings.MONGODB_URI,
        maxPoolSize=10,
        minPoolSize=1,
        serverSelectionTimeoutMS=5000,
    )
    _database = _client[settings.MONGODB_DB_NAME]

    # Verify connection by pinging the server
    try:
        await _client.admin.command("ping")
        logger.info(
            f"Connected to MongoDB database: {settings.MONGODB_DB_NAME}"
        )
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise


async def close_mongodb_connection() -> None:
    """
    Close the MongoDB connection.

    Called during FastAPI lifespan shutdown.
    """
    global _client, _database

    if _client is not None:
        _client.close()
        _client = None
        _database = None
        logger.info("MongoDB connection closed.")


async def get_database() -> AsyncIOMotorDatabase:
    """
    Get the MongoDB database instance.

    Used as a FastAPI dependency to inject the database
    into route handlers and services.

    Returns:
        The AsyncIOMotorDatabase instance.

    Raises:
        RuntimeError: If the database connection is not established.
    """
    if _database is None:
        raise RuntimeError(
            "Database not initialized. Ensure connect_to_mongodb() "
            "was called during application startup."
        )
    return _database


def get_client() -> Optional[AsyncIOMotorClient]:
    """
    Get the raw MongoDB client.

    Primarily used for administrative operations and health checks.

    Returns:
        The AsyncIOMotorClient instance, or None if not connected.
    """
    return _client
