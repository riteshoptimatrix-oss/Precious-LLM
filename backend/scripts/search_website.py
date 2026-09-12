"""
Precious AI — CLI: Search Website Knowledge

Tests website knowledge retrieval directly from the CLI.
Prints scored chunks with their source URLs, sections, and content snippets.

Usage:
    python -m scripts.search_website "study visa Canada"
    python -m scripts.search_website "IELTS coaching fees" --top-k 5
"""

import argparse
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongodb, get_database, close_mongodb_connection
from app.website.services.website_knowledge_service import WebsiteKnowledgeService

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("search_website")


async def main(args: argparse.Namespace) -> None:
    await connect_to_mongodb()
    db = await get_database()

    try:
        service = WebsiteKnowledgeService(db)

        query = args.query
        top_k = args.top_k

        print(f"\n[SEARCH] Query: \"{query}\"  (top_k={top_k})")
        print("=" * 70)

        chunks = await service.search(query, top_k=top_k)

        if not chunks:
            print("  [WARNING] No relevant chunks found. Check that a crawl has been run and a version is active.")
            print("=" * 70)
            return

        for i, chunk in enumerate(chunks, start=1):
            title = chunk.get("title", "")
            section = chunk.get("section", "")
            url = chunk.get("canonical_url", "")
            content = chunk.get("content", "")
            page_type = chunk.get("page_type", "")

            # Truncate content for display
            snippet = content[:300].replace("\n", " ") + ("..." if len(content) > 300 else "")

            print(f"\n[{i}] {title}")
            print(f"    Section   : {section}")
            print(f"    Page Type : {page_type}")
            print(f"    URL       : {url}")
            print(f"    Content   : {snippet}")

        print("\n" + "=" * 70)

        # Also show formatted context block
        if args.show_context:
            print("\n[CONTEXT] Formatted Context Block:")
            print("-" * 70)
            context = await service.get_context_block(query, top_k=top_k)
            print(context or "[No context generated]")
            print("-" * 70)
    finally:
        await close_mongodb_connection()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test website knowledge retrieval from CLI.")
    parser.add_argument("query", type=str, help="Search query string.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results to return (default: 3).")
    parser.add_argument("--show-context", action="store_true", help="Print the formatted LLM context block.")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
