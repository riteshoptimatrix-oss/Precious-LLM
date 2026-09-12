"""
Precious Edu LLM — Test Configuration & Shared Fixtures

Provides common test fixtures:
- Async HTTP client for API testing
- Test settings & isolated test DB
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.config import Settings, get_settings
from app.db.mongodb import connect_to_mongodb, close_mongodb_connection, get_client, get_database


@pytest.fixture(autouse=True)
def override_settings(monkeypatch):
    """Override application settings for testing."""
    get_settings.cache_clear()
    test_settings = Settings(
        APP_ENV="testing",
        APP_DEBUG=False,
        MONGODB_URI="mongodb://127.0.0.1:27017",
        MONGODB_DB_NAME="precious_ai_test",
        RATE_LIMIT_PER_MINUTE=1000,
        LOG_LEVEL="WARNING",
    )
    app.dependency_overrides[get_settings] = lambda: test_settings
    yield test_settings
    get_settings.cache_clear()
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_db():
    """Provide an isolated MongoDB test database fixture."""
    await connect_to_mongodb()
    client_conn = get_client()
    if client_conn:
        db = client_conn["precious_ai_test"]
        # Clear dynamic conversation collections
        for col in ["messages", "sessions", "memories"]:
            await db[col].delete_many({})
    else:
        db = await get_database()
    yield db


@pytest_asyncio.fixture
async def async_client():
    """Provide an async HTTP client for testing endpoints."""
    await connect_to_mongodb()
    client_conn = get_client()
    if client_conn:
        db = client_conn["precious_ai_test"]
        # Clear dynamic conversation collections
        for col in ["messages", "sessions", "memories"]:
            await db[col].delete_many({})

        # Ensure structured_qa and website_chunks are populated in test DB
        if await db["structured_qa"].count_documents({}) == 0:
            src_db = client_conn["precious_edu_llm"]
            try:
                await src_db["structured_qa"].aggregate([
                    {"$match": {}},
                    {"$out": {"db": "precious_ai_test", "coll": "structured_qa"}}
                ]).to_list(length=1)
                await src_db["website_chunks"].aggregate([
                    {"$match": {}},
                    {"$out": {"db": "precious_ai_test", "coll": "website_chunks"}}
                ]).to_list(length=1)
            except Exception:
                pass

        # Create indexes
        from app.db.indexes import create_database_indexes
        await create_database_indexes(db)

    # Ensure LLM service is initialized
    from app.llm import get_llm_service
    llm = get_llm_service()
    if not llm.is_ready():
        llm.initialize()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    if client_conn:
        db = client_conn["precious_ai_test"]
        for col in ["messages", "sessions", "memories"]:
            await db[col].delete_many({})
    await close_mongodb_connection()
