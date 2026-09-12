import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongodb, get_database

async def main():
    await connect_to_mongodb()
    db = await get_database()

    pages = await db.website_pages.find({}).to_list(100)
    print(f"Total crawled pages: {len(pages)}")
    for p in pages:
        print(f"Page URL: {p.get('url')} | Title: {p.get('title')} | Type: {p.get('page_type')}")

    chunks = await db.website_chunks.find({}).to_list(200)
    print(f"\nTotal active chunks: {len(chunks)}")
    visa_chunks = [c for c in chunks if any(w in c.get('content', '').lower() or w in c.get('section', '').lower() for w in ['visa', 'study', 'abroad', 'australia', 'canada', 'uk', 'usa'])]
    print(f"Visa/Study/Abroad Chunks found: {len(visa_chunks)}")
    for i, c in enumerate(visa_chunks, 1):
        print(f"[{i}] URL: {c.get('canonical_url')} | Section: '{c.get('section')}'")
        print(f"    Snippet: {c.get('content')[:150]}...\n")

if __name__ == "__main__":
    asyncio.run(main())
