"""
Precious Edu LLM — Structured Q&A Importer Script

Recursively scans backend/data/data/ and imports structured Q&A records
into MongoDB collection structured_qa idempotently.

Usage:
    python scripts/import_structured_qa.py
    python scripts/import_structured_qa.py --stats-only
"""

import argparse
import asyncio
import os
import sys

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongodb, close_mongodb_connection, get_database
from app.db.indexes import create_database_indexes
from app.knowledge.importer import StructuredQAImporter


async def main():
    parser = argparse.ArgumentParser(description="Import structured Q&A dataset into MongoDB.")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Inspect dataset files and print statistics without writing to MongoDB."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional custom path to data/data directory."
    )
    args = parser.parse_args()

    print("=" * 65)
    print("  PRECIOUS EDU LLM — STRUCTURED Q&A IMPORTER")
    print("=" * 65)

    await connect_to_mongodb()
    db = await get_database()

    importer = StructuredQAImporter(db=db, data_dir=args.data_dir)
    print(f"Target Data Directory : {importer.data_dir}")
    print(f"Target Database       : {getattr(db, 'name', 'precious_edu_llm')}")
    print(f"Target Collection     : {importer.collection_name}")
    print("-" * 65)

    if args.stats_only:
        print("Gathering dataset statistics...")
        stats = await importer.get_statistics()
        print(f"\nFiles Discovered      : {stats['total_files']}")
        print(f"Records Discovered    : {stats['total_records_discovered']}")
        print(f"Unique Records        : {stats['unique_records']}")
        print(f"Duplicate Records     : {stats['duplicates']}")
        print(f"Invalid Records       : {stats['invalid_records']}")
        print(f"Unique Intents Count  : {stats['unique_intents_count']}")
        print(f"MongoDB Current Count : {stats['mongodb_collection_count']}")
        print("\n--- Files Breakdown ---")
        for f in stats["files_breakdown"]:
            print(f"  {f['file']:40} | Records: {f['records']:5} | Unique Intents: {f['unique_intents']:3}")
        print("\n--- Top 10 Intents ---")
        for intent, count in stats["top_intents"][:10]:
            print(f"  {intent:35}: {count}")
    else:
        print("Creating/Verifying MongoDB indexes first...")
        await create_database_indexes(db)

        print("\nStarting idempotent bulk import...")
        report = await importer.run_import()

        # Check total count in DB after import
        final_count = await db[importer.collection_name].count_documents({})

        print("\n" + "=" * 65)
        print("  IMPORT EXECUTION REPORT")
        print("=" * 65)
        print(f"  Files Discovered      : {report.files_discovered}")
        print(f"  Files Processed       : {report.files_processed}")
        print(f"  Records Discovered    : {report.records_discovered}")
        print(f"  Records Imported (New): {report.records_imported}")
        print(f"  Records Updated       : {report.records_updated}")
        print(f"  Duplicates Skipped    : {report.duplicates_skipped}")
        print(f"  Invalid Records       : {report.invalid_records}")
        print(f"  Errors Encountered    : {len(report.errors)}")
        print(f"  MongoDB Database      : {report.database}")
        print(f"  MongoDB Collection    : {report.collection}")
        print(f"  Total In Collection   : {final_count}")
        print("=" * 65)

        if report.errors:
            print("\nErrors logged:")
            for err in report.errors:
                print(f"  - {err}")

    await close_mongodb_connection()
    print("\nOperation completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
