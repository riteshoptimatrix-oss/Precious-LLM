"""
Precious AI — CLI: Activate Website Version

Activates a specific website knowledge version or rolls back to a previous one.
Lists all available versions when called with --list.

Usage:
    python -m scripts.activate_website_version --list
    python -m scripts.activate_website_version web-v1726123456
    python -m scripts.activate_website_version web-v1726000000 --rollback
"""

import argparse
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongodb, get_database, close_mongodb_connection
from app.website.processing.versioning import WebsiteVersionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("activate_website_version")


async def main(args: argparse.Namespace) -> None:
    await connect_to_mongodb()
    db = await get_database()

    try:
        manager = WebsiteVersionManager(db)

        if args.list:
            versions = await manager.list_all_versions()
            print(f"\n{'=' * 60}")
            print(f"  All Recorded Website Versions ({len(versions)})")
            print(f"{'=' * 60}")
            for v in versions:
                active_marker = " ◄ ACTIVE" if v.get("is_active") else ""
                print(
                    f"  {v.get('website_version')}"
                    f" | pages={v.get('pages_count', 0)}"
                    f" | chunks={v.get('chunks_count', 0)}"
                    f" | status={v.get('status')}"
                    f" | created={v.get('created_at', 'N/A')[:19]}"
                    f"{active_marker}"
                )
            print(f"{'=' * 60}")
            return

        if not args.version_id:
            print("Error: provide a VERSION_ID or use --list.")
            sys.exit(1)

        try:
            if args.rollback:
                result = await manager.rollback_to_version(args.version_id)
                action = "Rollback"
            else:
                result = await manager.activate_version(args.version_id)
                action = "Activation"

            print(f"\n{'=' * 60}")
            print(f"  {action} — {result.get('status')}")
            print(f"{'=' * 60}")
            print(f"  Version        : {result.get('website_version')}")
            print(f"  Pages Activated: {result.get('pages_activated')}")
            print(f"  Chunks Activated: {result.get('chunks_activated')}")
            print(f"  Activated At   : {result.get('activated_at')}")
            print(f"{'=' * 60}")

        except ValueError as e:
            print(f"\nError: {e}")
            sys.exit(1)
    finally:
        await close_mongodb_connection()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Activate or rollback a website knowledge version."
    )
    parser.add_argument("version_id", nargs="?", default=None, help="Version ID to activate.")
    parser.add_argument("--rollback", action="store_true", help="Mark operation as rollback in logs.")
    parser.add_argument("--list", action="store_true", help="List all available versions.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
