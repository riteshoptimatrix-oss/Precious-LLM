"""
Precious AI — CLI: Website Crawler

Runs a full or dry-run crawl of https://www.preciousedu.in/
and indexes the results into MongoDB.

Usage:
    python -m scripts.crawl_website
    python -m scripts.crawl_website --dry-run
    python -m scripts.crawl_website --max-pages 100 --max-depth 3
"""

import argparse
import asyncio
import logging
import sys
import os

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongodb, get_database, close_mongodb_connection
from app.website.services.crawl_service import CrawlService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("crawl_website")


async def main(args: argparse.Namespace) -> None:
    logger.info("Connecting to MongoDB...")
    await connect_to_mongodb()
    db = await get_database()

    try:
        service = CrawlService(
            db=db,
            base_url="https://www.preciousedu.in/",
            max_depth=args.max_depth,
            max_pages=args.max_pages,
            delay_seconds=args.delay,
            concurrency=args.concurrency,
        )

        logger.info(
            f"{'[DRY-RUN] ' if args.dry_run else ''}Starting crawl "
            f"(max_pages={args.max_pages}, max_depth={args.max_depth}, "
            f"delay={args.delay}s, concurrency={args.concurrency})"
        )

        summary = await service.run_crawl(dry_run=args.dry_run)

        print("\n" + "=" * 60)
        print(f"  Crawl Summary — {summary.status}")
        print("=" * 60)
        print(f"  Crawl ID        : {summary.crawl_id}")
        print(f"  Version         : {summary.website_version}")
        print(f"  Pages Discovered: {summary.pages_discovered}")
        print(f"  Pages Fetched   : {summary.pages_fetched}")
        print(f"  Pages New       : {summary.pages_new}")
        print(f"  Pages Failed    : {summary.pages_failed}")
        print(f"  Chunks Created  : {summary.chunks_created}")
        print(f"  Started At      : {summary.started_at}")
        print(f"  Completed At    : {summary.completed_at}")

        if summary.errors:
            print(f"\n  Errors ({len(summary.errors)}):")
            for err in summary.errors[:10]:
                print(f"    - {err}")

        print("=" * 60)

        if summary.status not in ("COMPLETED", "DRY_RUN_COMPLETE"):
            sys.exit(1)
    finally:
        await close_mongodb_connection()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl https://www.preciousedu.in/ and index content into MongoDB."
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover URLs without writing to MongoDB.")
    parser.add_argument("--max-pages", type=int, default=500, help="Maximum pages to crawl (default: 500).")
    parser.add_argument("--max-depth", type=int, default=5, help="Maximum crawl depth (default: 5).")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between requests in seconds (default: 0.5).")
    parser.add_argument("--concurrency", type=int, default=2, help="Max concurrent connections (default: 2).")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
