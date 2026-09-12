"""
Precious AI — Semantic Section Website Chunker

Splits parsed web pages into context-rich section chunks preserving source URL and metadata.
"""

from typing import Dict, List, Any
from app.website.processing.hasher import ContentHasher


class WebsiteChunker:
    """
    Semantic website document chunker.
    """

    def __init__(self, max_words_per_chunk: int = 250, overlap_words: int = 25):
        self.max_words = max_words_per_chunk
        self.overlap = overlap_words

    def create_chunks(self, parsed_page: Dict[str, Any], website_version: str) -> List[Dict[str, Any]]:
        """
        Chunks parsed page sections into chunk dictionaries.
        """
        canonical_url = parsed_page.get("url", "")
        title = parsed_page.get("title", "Precious Education")
        page_type = parsed_page.get("page_type", "other")
        sections = parsed_page.get("sections", [])

        chunks = []
        chunk_idx = 0

        for sec in sections:
            sec_heading = sec.get("heading", title)
            sec_content = sec.get("content", "").strip()

            if not sec_content:
                continue

            words = sec_content.split()
            if len(words) <= self.max_words:
                c_hash = ContentHasher.compute_hash(sec_content)
                chunks.append({
                    "canonical_url": canonical_url,
                    "title": title,
                    "page_type": page_type,
                    "section": sec_heading,
                    "chunk_index": chunk_idx,
                    "content": sec_content,
                    "content_hash": c_hash,
                    "website_version": website_version,
                    "active": True
                })
                chunk_idx += 1
            else:
                # Sliding word window
                i = 0
                while i < len(words):
                    sub_words = words[i:i + self.max_words]
                    chunk_text = " ".join(sub_words)
                    c_hash = ContentHasher.compute_hash(chunk_text)

                    chunks.append({
                        "canonical_url": canonical_url,
                        "title": title,
                        "page_type": page_type,
                        "section": sec_heading,
                        "chunk_index": chunk_idx,
                        "content": chunk_text,
                        "content_hash": c_hash,
                        "website_version": website_version,
                        "active": True
                    })
                    chunk_idx += 1
                    i += (self.max_words - self.overlap)

        return chunks
