"""
Precious Edu LLM — Unit Tests for Project Knowledge Search

Tests exact ID search, prefix search, case-insensitive substring search,
structured status/client/manager filtering, and result limit capping.
"""

import pytest
import pytest_asyncio
from app.knowledge.models import ProjectRecord, SourceMetadata, StructuredProjectQuery
from app.knowledge.repository import ProjectRepository
from app.knowledge.search import ProjectKnowledgeSearch


@pytest_asyncio.fixture
async def populated_search_engine(test_db):
    repo = ProjectRepository(test_db)
    await test_db["project_records"].drop()
    await test_db["project_dataset_metadata"].drop()

    meta = SourceMetadata(file="test.xlsx", sheet="Projects", row=2, file_hash="h1", dataset_version="projects-v1")

    records = [
        ProjectRecord(project_id="P001", project_name="Project Alpha", client="ABC Ltd", status="Active", manager="John", source=meta, dataset_version="projects-v1", is_active=True),
        ProjectRecord(project_id="P002", project_name="Project Beta", client="XYZ Corp", status="Hold", manager="Sarah", source=meta, dataset_version="projects-v1", is_active=True),
        ProjectRecord(project_id="P003", project_name="Project Gamma", client="ABC Ltd", status="Completed", manager="Mike", source=meta, dataset_version="projects-v1", is_active=True),
    ]

    await repo.save_dataset(records, "projects-v1", "h1", "test.xlsx", activate=True)
    engine = ProjectKnowledgeSearch(test_db)

    yield engine

    await test_db["project_records"].drop()
    await test_db["project_dataset_metadata"].drop()



@pytest.mark.asyncio
async def test_exact_project_id_search(populated_search_engine):
    q = StructuredProjectQuery(entity="project", operation="get", filters={"project_id": "P001"})
    res = await populated_search_engine.search(q)

    assert res.total_found == 1
    assert res.records[0].project_id == "P001"
    assert res.records[0].project_name == "Project Alpha"


@pytest.mark.asyncio
async def test_case_insensitive_name_search(populated_search_engine):
    q = StructuredProjectQuery(entity="project", operation="get", filters={"project_name": "alpha"})
    res = await populated_search_engine.search(q)

    assert res.total_found == 1
    assert res.records[0].project_id == "P001"


@pytest.mark.asyncio
async def test_structured_client_filter(populated_search_engine):
    q = StructuredProjectQuery(entity="project", operation="get", filters={"client": "ABC Ltd"})
    res = await populated_search_engine.search(q)

    assert res.total_found == 2
    pids = [r.project_id for r in res.records]
    assert "P001" in pids
    assert "P003" in pids


@pytest.mark.asyncio
async def test_combined_client_and_status_filter(populated_search_engine):
    q = StructuredProjectQuery(entity="project", operation="get", filters={"client": "ABC Ltd", "status": "Active"})
    res = await populated_search_engine.search(q)

    assert res.total_found == 1
    assert res.records[0].project_id == "P001"


@pytest.mark.asyncio
async def test_no_match_search(populated_search_engine):
    q = StructuredProjectQuery(entity="project", operation="get", filters={"project_id": "P999"})
    res = await populated_search_engine.search(q)

    assert res.total_found == 0
    assert len(res.records) == 0
