"""
Precious AI — CLI: Incremental Website Refresh

Runs an incremental refresh: only re-indexes pages that have changed
since the last crawl (based on SHA-256 content hash comparison).

Usage:
    python -m scripts.refresh_website
    python -m scripts.refresh_website --rollback web-v1726123456
"""

import argparse
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongodb, get_database, close_mongodb_connection
from app.website.services.website_refresh_service import WebsiteRefreshService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("refresh_website")


async def main(args: argparse.Namespace) -> None:
    logger.info("Connecting to MongoDB...")
    await connect_to_mongodb()
    db = await get_database()

    try:
        if args.rollback:
            service = WebsiteRefreshService(db=db)
            logger.info(f"Rolling back to version: {args.rollback}")
            result = await service.rollback_to_version(args.rollback)
            print("\n" + "=" * 60)
            print(f"  Rollback Result — {result.get('status')}")
            print("=" * 60)
            print(f"  Version        : {result.get('website_version')}")
            print(f"  Pages Activated: {result.get('pages_activated')}")
            print(f"  Chunks Activated: {result.get('chunks_activated')}")
            print(f"  Activated At   : {result.get('activated_at')}")
            print("=" * 60)
            return

        service = WebsiteRefreshService(
            db=db,
            base_url="https://www.preciousedu.in/",
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            delay_seconds=args.delay,
            concurrency=args.concurrency,
        )

        logger.info("Starting incremental website refresh...")
        summary = await service.run_refresh()

        print("\n" + "=" * 60)
        print(f"  Refresh Summary — {summary.status}")
        print("=" * 60)
        print(f"  Crawl ID        : {summary.crawl_id}")
        print(f"  Version         : {summary.website_version}")
        print(f"  Pages New       : {summary.pages_new}")
        print(f"  Pages Changed   : {summary.pages_changed}")
        print(f"  Pages Unchanged : {summary.pages_unchanged}")
        print(f"  Pages Removed   : {summary.pages_removed}")
        print(f"  Pages Failed    : {summary.pages_failed}")
        print(f"  Chunks Created  : {summary.chunks_created}")
        print(f"  Completed At    : {summary.completed_at}")
        print("=" * 60)

        if summary.status != "COMPLETED":
            sys.exit(1)
    finally:
        await close_mongodb_connection()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally refresh the website knowledge index."
    )
    parser.add_argument("--rollback", type=str, default=None, metavar="VERSION_ID",
                        help="Roll back to a specific version ID instead of crawling.")
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--concurrency", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
