"""
Precious Edu LLM — Excel Loader & Inspection Module

Reads and inspects `.xlsx` and `.xls` workbooks using openpyxl.
Ensures the original Excel file remains strictly immutable.
Extracts sheet structures, headers, row metadata, merged cell information,
and raw dictionary records with row-level provenance.
"""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

from app.knowledge.exceptions import ExcelIngestionError, InvalidWorkbookError
from app.knowledge.models import IngestionReport

logger = logging.getLogger(__name__)


class ExcelLoader:
    """
    Safe, read-only loader for Excel workbooks.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path).resolve()
        if not self.file_path.exists():
            raise ExcelIngestionError(f"Excel file not found: {self.file_path}")

        if not self.file_path.suffix.lower() in [".xlsx", ".xls"]:
            raise InvalidWorkbookError(f"Unsupported file format: {self.file_path.suffix}. Expected .xlsx or .xls")

    def compute_sha256(self) -> str:
        """Compute deterministic SHA-256 hash of the source Excel file."""
        hasher = hashlib.sha256()
        with open(self.file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def inspect(self, dataset_version: str = "projects-v1") -> IngestionReport:
        """
        Inspect workbook structure, sheet counts, row counts, headers, and metadata
        without mutating database or source files.
        """
        file_hash = self.compute_sha256()
        file_size = self.file_path.stat().st_size

        try:
            wb = openpyxl.load_workbook(self.file_path, data_only=True)
        except Exception as e:
            raise InvalidWorkbookError(f"Failed to open Excel workbook: {e}") from e

        sheet_names = wb.sheetnames
        sheet_details: Dict[str, Dict[str, Any]] = {}
        total_rows_parsed = 0

        for sheet_name in sheet_names:
            ws: Worksheet = wb[sheet_name]
            max_row = ws.max_row or 0
            max_col = ws.max_column or 0
            merged_cells_count = len(ws.merged_cells.ranges)

            # Detect headers from row 1
            headers = []
            if max_row > 0:
                for col in range(1, max_col + 1):
                    val = ws.cell(row=1, column=col).value
                    headers.append(str(val).strip() if val is not None else f"Column_{col}")

            # Count empty vs populated rows
            empty_rows = 0
            populated_rows = 0
            for r in range(2, max_row + 1):
                row_vals = [ws.cell(row=r, column=c).value for c in range(1, max_col + 1)]
                if all(v is None or str(v).strip() == "" for v in row_vals):
                    empty_rows += 1
                else:
                    populated_rows += 1

            total_rows_parsed += populated_rows

            sheet_details[sheet_name] = {
                "max_row": max_row,
                "max_column": max_col,
                "header_count": len(headers),
                "headers": headers,
                "populated_rows": populated_rows,
                "empty_rows": empty_rows,
                "merged_cells_count": merged_cells_count,
            }

        wb.close()

        report = IngestionReport(
            filename=self.file_path.name,
            file_size_bytes=file_size,
            file_hash=file_hash,
            dataset_version=dataset_version,
            sheet_count=len(sheet_names),
            sheet_names=sheet_names,
            sheet_details=sheet_details,
            total_rows_parsed=total_rows_parsed,
        )

        return report

    def parse_rows(self, dataset_version: str = "projects-v1") -> Tuple[List[Dict[str, Any]], IngestionReport]:
        """
        Parse all non-empty rows across workbook sheets into raw dictionary records
        with source provenance metadata.
        """
        report = self.inspect(dataset_version=dataset_version)
        file_hash = report.file_hash

        try:
            wb = openpyxl.load_workbook(self.file_path, data_only=True)
        except Exception as e:
            raise InvalidWorkbookError(f"Failed to load Excel workbook for row parsing: {e}") from e

        raw_records: List[Dict[str, Any]] = []

        for sheet_name in wb.sheetnames:
            ws: Worksheet = wb[sheet_name]
            max_row = ws.max_row or 0
            max_col = ws.max_column or 0

            if max_row < 2:
                continue

            # Read headers
            headers = []
            for col in range(1, max_col + 1):
                val = ws.cell(row=1, column=col).value
                headers.append(str(val).strip() if val is not None else f"Column_{col}")

            for row_idx in range(2, max_row + 1):
                row_vals = [ws.cell(row=row_idx, column=c).value for c in range(1, max_col + 1)]

                # Skip completely empty rows
                if all(v is None or str(v).strip() == "" for v in row_vals):
                    continue

                row_dict = {}
                for h, val in zip(headers, row_vals):
                    row_dict[h] = val

                record_data = {
                    "raw_data": row_dict,
                    "source": {
                        "file": self.file_path.name,
                        "sheet": sheet_name,
                        "row": row_idx,
                        "file_hash": file_hash,
                        "dataset_version": dataset_version,
                    }
                }
                raw_records.append(record_data)

        wb.close()
        return raw_records, report
