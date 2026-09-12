"""
Precious Edu LLM — CLI Project Data Ingestion Script

Ingests Excel workbook into MongoDB project knowledge store with validation,
provenance tracking, dataset versioning, and activation.

Usage:
    python -m scripts.ingest_projects --file data/projects/raw/projects.xlsx
"""

import argparse
import asyncio
import sys
from pathlib import Path

from app.db.mongodb import connect_to_mongodb, close_mongodb_connection, get_database
from app.knowledge.excel_loader import ExcelLoader
from app.knowledge.exceptions import ExcelIngestionError, InvalidWorkbookError
from app.knowledge.mapper import ColumnMapper
from app.knowledge.normalizer import DataNormalizer
from app.knowledge.repository import ProjectRepository
from app.knowledge.validator import RecordValidator


async def async_ingest(file_path: Path):
    print("=" * 60)
    print("PRECIOUS AI — PROJECT DATA INGESTION PIPELINE")
    print("=" * 60)
    print(f"Source File: {file_path}")

    # 1. Connect to MongoDB
    await connect_to_mongodb()
    db = await get_database()
    repo = ProjectRepository(db)

    # 2. Parse workbook
    loader = ExcelLoader(file_path)
    report = loader.inspect()
    file_hash = report.file_hash

    # 3. Check hash idempotency
    existing_meta = await repo.get_metadata_by_hash(file_hash)
    if existing_meta and existing_meta.get("is_active"):
        print(f"NOTICE: Workbook has already been ingested under dataset version '{existing_meta.get('dataset_version')}'.")
        print("Skipping re-ingestion as file hash is unchanged.")
        await close_mongodb_connection()
        sys.exit(0)

    dataset_version = await repo.get_next_version_id()
    print(f"Target Dataset Version: {dataset_version}")
    print(f"Source SHA-256 Hash: {file_hash}")

    raw_records, _ = loader.parse_rows(dataset_version=dataset_version)

    # 4. Map, Normalize, Validate
    mapper = ColumnMapper()
    normalizer = DataNormalizer()
    validator = RecordValidator()

    valid_records, invalid_reports = validator.validate_records(
        raw_parsed_records=raw_records,
        mapper=mapper,
        normalizer=normalizer
    )

    print(f"Parsed Rows: {len(raw_records)}")
    print(f"Valid Canonical Records: {len(valid_records)}")
    print(f"Invalid Rows Reported: {len(invalid_reports)}")

    if invalid_reports:
        print("\n--- VALIDATION ERROR REPORT ---")
        for report in invalid_reports:
            src = report.get("source", {})
            errs = ", ".join(report.get("errors", []))
            print(f"Sheet: {src.get('sheet')}, Row: {src.get('row')} -> Errors: {errs}")

    if not valid_records:
        print("\nERROR: No valid project records produced. Dataset version creation aborted.")
        await close_mongodb_connection()
        sys.exit(1)

    # 5. Save & Activate in MongoDB
    saved_version = await repo.save_dataset(
        records=valid_records,
        dataset_version=dataset_version,
        file_hash=file_hash,
        source_file=file_path.name,
        activate=True
    )

    print("-" * 60)
    print(f"SUCCESS: Ingested {len(valid_records)} project records into MongoDB.")
    print(f"Active Dataset Version: {saved_version}")
    print("=" * 60)

    await close_mongodb_connection()


def main():
    parser = argparse.ArgumentParser(description="Ingest an Excel workbook into Project Knowledge Engine.")
    parser.add_argument("--file", "-f", required=True, help="Path to Excel file (.xlsx or .xls)")
    args = parser.parse_args()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"ERROR: Target file does not exist: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        asyncio.run(async_ingest(file_path))
    except Exception as e:
        print(f"INGESTION ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
