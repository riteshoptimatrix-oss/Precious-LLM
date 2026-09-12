"""
Precious Edu LLM — Knowledge Indexer

Indexes parsed documents for efficient retrieval.
Supports building and searching a text index.
"""

import logging

logger = logging.getLogger(__name__)


class KnowledgeIndexer:
    """
    Builds and manages a search index over knowledge documents.

    Initial implementation: TF-IDF based index.
    Future: Embedding-based similarity search.
    """

    def __init__(self):
        self._index = None
        self._documents = []

    async def build_index(self, documents: list) -> None:
        """
        Build a search index from a list of documents.

        Args:
            documents: List of document dicts with 'content' and 'metadata'.
        """
        # TODO (Phase 13): Implement TF-IDF index
        raise NotImplementedError("KnowledgeIndexer.build_index — Phase 13")

    async def search(self, query: str, top_k: int = 3) -> list:
        """
        Search the index for documents relevant to a query.

        Args:
            query: Search query text.
            top_k: Maximum results to return.

        Returns:
            List of (document, score) tuples.
        """
        # TODO (Phase 13): Implement
        raise NotImplementedError("KnowledgeIndexer.search — Phase 13")
