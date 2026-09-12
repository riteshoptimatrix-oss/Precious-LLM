"""
Precious Edu LLM — Database Initialization Script

Standalone script to initialize MongoDB collections and indexes.
Run command:
    python scripts/init_database.py
"""

import sys
import os
import asyncio
import logging

# Ensure project root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import get_settings
from app.db.indexes import create_database_indexes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def init_db():
    settings = get_settings()
    logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}...")
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_DB_NAME]

    await db.command("ping")
    logger.info(f"Connected to database: {settings.MONGODB_DB_NAME}")

    await create_database_indexes(db)
    logger.info("Database initialization completed successfully.")
    client.close()


if __name__ == "__main__":
    asyncio.run(init_db())
