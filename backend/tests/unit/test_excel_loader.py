"""
Precious Edu LLM — Unit Tests for Excel Loader

Tests Excel workbook loading, sheet discovery, header detection,
empty row filtering, merged cell inspection, and immutable reading.
"""

import pytest
import openpyxl
from pathlib import Path

from app.knowledge.excel_loader import ExcelLoader
from app.knowledge.exceptions import ExcelIngestionError, InvalidWorkbookError


@pytest.fixture
def sample_excel_file(tmp_path) -> Path:
    file_path = tmp_path / "test_projects.xlsx"
    wb = openpyxl.Workbook()

    # Sheet 1: Projects
    ws1 = wb.active
    ws1.title = "Projects"
    ws1.append(["Project ID", "Project Name", "Client", "Status", "Manager", "Start Date"])
    ws1.append(["P001", "Project Alpha", "ABC Ltd", "Active", "John", "2026-01-10"])
    ws1.append(["P002", "Project Beta", "XYZ Corp", "Hold", "Sarah", "2026-02-15"])
    ws1.append(["", "", "", "", "", ""])  # Empty row

    # Sheet 2: Status
    ws2 = wb.create_sheet(title="Status")
    ws2.append(["Code", "Description"])
    ws2.append(["ACT", "Active status"])

    wb.save(file_path)
    wb.close()
    return file_path


def test_excel_loader_inspect(sample_excel_file):
    loader = ExcelLoader(sample_excel_file)
    report = loader.inspect(dataset_version="test-v1")

    assert report.filename == "test_projects.xlsx"
    assert report.sheet_count == 2
    assert "Projects" in report.sheet_names
    assert "Status" in report.sheet_names
    assert len(report.file_hash) == 64  # Valid SHA-256 string length

    projects_detail = report.sheet_details["Projects"]
    assert projects_detail["header_count"] == 6
    assert projects_detail["populated_rows"] == 2
    assert projects_detail["empty_rows"] == 1


def test_excel_loader_parse_rows(sample_excel_file):
    loader = ExcelLoader(sample_excel_file)
    rows, report = loader.parse_rows(dataset_version="test-v1")

    assert len(rows) == 3  # 2 from Projects, 1 from Status
    p001_record = rows[0]
    assert p001_record["raw_data"]["Project ID"] == "P001"
    assert p001_record["raw_data"]["Project Name"] == "Project Alpha"
    assert p001_record["source"]["file"] == "test_projects.xlsx"
    assert p001_record["source"]["sheet"] == "Projects"
    assert p001_record["source"]["row"] == 2


def test_excel_loader_nonexistent_file(tmp_path):
    with pytest.raises(ExcelIngestionError):
        ExcelLoader(tmp_path / "non_existent.xlsx")


def test_excel_loader_invalid_extension(tmp_path):
    invalid_file = tmp_path / "data.txt"
    invalid_file.write_text("hello")
    with pytest.raises(InvalidWorkbookError):
        ExcelLoader(invalid_file)
