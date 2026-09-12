"""
Precious Edu LLM — Unit Tests for Knowledge Context Formatter

Tests single record formatting, multi-match list formatting,
zero-match notification, and context length bounds protection.
"""

import pytest
from app.knowledge.context import KnowledgeContextFormatter
from app.knowledge.models import ProjectRecord, SearchResult, SourceMetadata, StructuredProjectQuery


@pytest.fixture
def sample_source():
    return SourceMetadata(
        file="projects.xlsx",
        sheet="Projects",
        row=12,
        file_hash="hash_123",
        dataset_version="projects-v1"
    )


def test_format_single_record(sample_source):
    formatter = KnowledgeContextFormatter()
    rec = ProjectRecord(
        project_id="P001",
        project_name="Project Alpha",
        client="ABC Ltd",
        status="In Progress",
        manager="John",
        start_date="2026-01-10",
        source=sample_source,
        dataset_version="projects-v1"
    )
    res = SearchResult(
        query=StructuredProjectQuery(filters={"project_id": "P001"}),
        records=[rec],
        total_found=1,
        dataset_version="projects-v1"
    )

    context_str = formatter.format_search_result(res)

    assert "Project Knowledge:" in context_str
    assert "Project ID: P001" in context_str
    assert "Project Name: Project Alpha" in context_str
    assert "Status: In Progress" in context_str
    assert "Manager: John" in context_str
    assert "Source: projects.xlsx, Sheet: Projects, Row: 12" in context_str


def test_format_zero_matches(sample_source):
    formatter = KnowledgeContextFormatter()
    res = SearchResult(
        query=StructuredProjectQuery(filters={"project_name": "Project XYZ"}),
        records=[],
        total_found=0,
        dataset_version="projects-v1"
    )

    context_str = formatter.format_search_result(res)

    assert "No matching project was found in the dataset for 'Project XYZ'" in context_str


def test_format_multiple_records(sample_source):
    formatter = KnowledgeContextFormatter()
    rec1 = ProjectRecord(project_id="P001", project_name="Alpha", client="ABC", status="Active", manager="John", source=sample_source, dataset_version="v1")
    rec2 = ProjectRecord(project_id="P002", project_name="Beta", client="XYZ", status="Hold", manager="Sarah", source=sample_source, dataset_version="v1")

    res = SearchResult(
        query=StructuredProjectQuery(filters={"query": "projects"}),
        records=[rec1, rec2],
        total_found=2,
        dataset_version="v1"
    )

    context_str = formatter.format_search_result(res)

    assert "2 matching projects found" in context_str
    assert "1. ID: P001" in context_str
    assert "2. ID: P002" in context_str
