"""
Precious AI — SHA-256 Content Hasher for Website Change Detection
"""

import hashlib


class ContentHasher:
    """
    Computes deterministic cryptographic content hashes.
    """

    @staticmethod
    def compute_hash(text: str) -> str:
        """
        Returns SHA-256 hex digest of normalized string text.
        """
        if not text:
            return ""
        norm_text = " ".join(text.strip().split())
        return hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
