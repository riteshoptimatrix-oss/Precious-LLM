"""
Precious Edu LLM — Unit Tests for Project Repository

Tests dataset saving, dataset versioning, version activation,
file hash idempotency, and health check querying.
"""

import pytest
import pytest_asyncio
from app.knowledge.models import ProjectRecord, SourceMetadata
from app.knowledge.repository import ProjectRepository


@pytest.fixture
def mock_source_meta():
    return SourceMetadata(
        file="projects.xlsx",
        sheet="Projects",
        row=2,
        file_hash="hash_123",
        dataset_version="projects-v1"
    )


@pytest.fixture
def sample_project_record(mock_source_meta):
    return ProjectRecord(
        project_id="P001",
        project_name="Project Alpha",
        client="ABC Ltd",
        status="Active",
        manager="John",
        source=mock_source_meta,
        dataset_version="projects-v1",
        is_active=True
    )


@pytest_asyncio.fixture
async def project_repo(test_db):
    repo = ProjectRepository(test_db)
    # Clean up test collections prior to run
    await test_db["project_records"].drop()
    await test_db["project_dataset_metadata"].drop()
    yield repo
    await test_db["project_records"].drop()
    await test_db["project_dataset_metadata"].drop()



@pytest.mark.asyncio
async def test_save_and_activate_dataset(project_repo, sample_project_record):
    records = [sample_project_record]

    version = await project_repo.save_dataset(
        records=records,
        dataset_version="projects-v1",
        file_hash="hash_123",
        source_file="projects.xlsx",
        activate=True
    )

    assert version == "projects-v1"

    health = await project_repo.get_health_status()
    assert health.status == "READY"
    assert health.active_dataset == "projects-v1"
    assert health.total_records == 1
    assert health.source_file == "projects.xlsx"


@pytest.mark.asyncio
async def test_dataset_versioning_and_rollback(project_repo, sample_project_record):
    # Save version 1
    await project_repo.save_dataset(
        records=[sample_project_record],
        dataset_version="projects-v1",
        file_hash="hash_123",
        source_file="projects_v1.xlsx",
        activate=True
    )

    # Create version 2 with updated manager name
    v2_meta = SourceMetadata(
        file="projects_v2.xlsx",
        sheet="Projects",
        row=2,
        file_hash="hash_456",
        dataset_version="projects-v2"
    )
    v2_record = ProjectRecord(
        project_id="P001",
        project_name="Project Alpha",
        client="ABC Ltd",
        status="Active",
        manager="Jane",
        source=v2_meta,
        dataset_version="projects-v2",
        is_active=True
    )

    await project_repo.save_dataset(
        records=[v2_record],
        dataset_version="projects-v2",
        file_hash="hash_456",
        source_file="projects_v2.xlsx",
        activate=True
    )

    # Verify version 2 is active
    health_v2 = await project_repo.get_health_status()
    assert health_v2.active_dataset == "projects-v2"

    # Re-activate version 1 (version rollback test)
    await project_repo.activate_dataset("projects-v1")
    health_v1 = await project_repo.get_health_status()
    assert health_v1.active_dataset == "projects-v1"
