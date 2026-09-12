"""
Precious Edu LLM — MongoDB Index Initialization

Creates necessary collection indexes at startup or via database initialization scripts.
"""

import logging
from pymongo import ASCENDING, DESCENDING
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.collections import SESSIONS_COLLECTION, MESSAGES_COLLECTION, MEMORIES_COLLECTION, STRUCTURED_QA_COLLECTION

logger = logging.getLogger(__name__)


async def create_database_indexes(db: AsyncIOMotorDatabase) -> None:
    """
    Ensure required MongoDB indexes exist.

    Indexes created:
    - sessions: session_id (unique), updated_at (descending)
    - messages: session_id + created_at (compound)
    - memories: session_id + key (unique compound), session_id + updated_at
    - structured_qa: record_hash (unique), intent, normalized_input, category, active, keywords
    """
    logger.info("Initializing MongoDB indexes...")

    # Sessions collection indexes
    sessions = db[SESSIONS_COLLECTION]
    await sessions.create_index([("session_id", ASCENDING)], unique=True, name="idx_session_id_unique")
    await sessions.create_index([("updated_at", DESCENDING)], name="idx_sessions_updated_at")

    # Messages collection indexes
    messages = db[MESSAGES_COLLECTION]
    await messages.create_index(
        [("session_id", ASCENDING), ("created_at", ASCENDING)],
        name="idx_messages_session_created"
    )

    # Memories collection indexes
    memories = db[MEMORIES_COLLECTION]
    await memories.create_index(
        [("session_id", ASCENDING), ("key", ASCENDING)],
        unique=True,
        name="idx_memories_session_key_unique"
    )
    await memories.create_index(
        [("session_id", ASCENDING), ("updated_at", DESCENDING)],
        name="idx_memories_session_updated"
    )

    # Structured QA collection indexes
    structured_qa = db[STRUCTURED_QA_COLLECTION]
    await structured_qa.create_index([("record_hash", ASCENDING)], unique=True, name="idx_sqa_record_hash_unique")
    await structured_qa.create_index([("intent", ASCENDING)], name="idx_sqa_intent")
    await structured_qa.create_index([("normalized_input", ASCENDING)], name="idx_sqa_normalized_input")
    await structured_qa.create_index([("category", ASCENDING)], name="idx_sqa_category")
    await structured_qa.create_index([("source", ASCENDING)], name="idx_sqa_source")
    await structured_qa.create_index([("active", ASCENDING)], name="idx_sqa_active")
    await structured_qa.create_index([("keywords", ASCENDING)], name="idx_sqa_keywords")
    await structured_qa.create_index(
        [("active", ASCENDING), ("intent", ASCENDING)],
        name="idx_sqa_active_intent"
    )
    await structured_qa.create_index(
        [("active", ASCENDING), ("keywords", ASCENDING)],
        name="idx_sqa_active_keywords"
    )

    logger.info("MongoDB indexes successfully created/verified.")

