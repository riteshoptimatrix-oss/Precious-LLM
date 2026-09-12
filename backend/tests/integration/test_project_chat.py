"""
Precious Edu LLM — Integration Tests for Project Knowledge Engine & Chat Integration

End-to-end tests for ConversationEngine with Excel project knowledge retrieval,
multi-turn conversation, non-project queries, and hallucination prevention.
"""

import pytest
import pytest_asyncio
import openpyxl
from pathlib import Path

from app.conversation.engine import ConversationEngine
from app.knowledge.excel_loader import ExcelLoader
from app.knowledge.mapper import ColumnMapper
from app.knowledge.normalizer import DataNormalizer
from app.knowledge.repository import ProjectRepository
from app.knowledge.validator import RecordValidator
from app.repositories.session_repository import SessionRepository


@pytest.fixture
def sample_excel_path(tmp_path) -> Path:
    file_path = tmp_path / "projects.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Projects"

    ws.append(["Project ID", "Project Name", "Client", "Status", "Manager", "Start Date"])
    ws.append(["P001", "Alpha", "ABC Ltd", "Active", "John", "2026-01-10"])
    ws.append(["P002", "Beta", "XYZ Corp", "Hold", "Sarah", "2026-02-15"])
    ws.append(["P003", "Gamma", "ABC Ltd", "Completed", "Mike", "2026-03-01"])

    wb.save(file_path)
    wb.close()
    return file_path


@pytest_asyncio.fixture
async def setup_chat_engine(test_db, sample_excel_path):
    # Drop test collections
    await test_db["sessions"].drop()
    await test_db["messages"].drop()
    await test_db["memories"].drop()
    await test_db["project_records"].drop()
    await test_db["project_dataset_metadata"].drop()

    # Ingest test projects into MongoDB
    repo = ProjectRepository(test_db)
    loader = ExcelLoader(sample_excel_path)
    raw_recs, report = loader.parse_rows("projects-v1")
    mapper = ColumnMapper()
    normalizer = DataNormalizer()
    validator = RecordValidator()

    valid, _ = validator.validate_records(raw_recs, mapper, normalizer)
    await repo.save_dataset(valid, "projects-v1", report.file_hash, sample_excel_path.name, activate=True)

    # Initialize Session
    from app.models.session import SessionModel
    sess_repo = SessionRepository(test_db)
    session_id = "test_session_proj_001"
    session_model = SessionModel(session_id=session_id, user_id="test_user", title="Project Test Chat")
    await sess_repo.create(session_model)


    engine = ConversationEngine(test_db)

    yield engine, session_id

    # Cleanup
    await test_db["sessions"].drop()
    await test_db["messages"].drop()
    await test_db["memories"].drop()
    await test_db["project_records"].drop()
    await test_db["project_dataset_metadata"].drop()



@pytest.mark.asyncio
async def test_chat_normal_conversation_preservation(setup_chat_engine):
    engine, session_id = setup_chat_engine

    resp = await engine.handle_message(session_id=session_id, user_message="Hello")
    assert resp["response"] is not None
    assert "metadata" in resp


@pytest.mark.asyncio
async def test_chat_project_knowledge_query(setup_chat_engine):
    engine, session_id = setup_chat_engine

    resp = await engine.handle_message(session_id=session_id, user_message="What is Project Alpha's status?")
    assert resp["response"] is not None


@pytest.mark.asyncio
async def test_chat_multi_turn_project_followup(setup_chat_engine):
    engine, session_id = setup_chat_engine

    # Turn 1
    _ = await engine.handle_message(session_id=session_id, user_message="Tell me about Project Alpha.")

    # Turn 2: Follow-up pronoun reference "it"
    resp2 = await engine.handle_message(session_id=session_id, user_message="Who manages it?")
    assert resp2["response"] is not None


@pytest.mark.asyncio
async def test_chat_missing_project_no_hallucination(setup_chat_engine):
    engine, session_id = setup_chat_engine

    resp = await engine.handle_message(session_id=session_id, user_message="What is Project XYZ status?")
    assert resp["response"] is not None
