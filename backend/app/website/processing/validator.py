"""
Precious AI — Website Knowledge Quality Validator
"""

from typing import List, Dict, Any, Tuple


class WebsiteQualityValidator:
    """
    Validates crawled website pages and chunks prior to activation.
    """

    @classmethod
    def validate_dataset(cls, pages: List[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates page list and chunk list for activation readiness.
        """
        errors = []
        warnings = []

        if not pages:
            errors.append("Dataset contains 0 pages.")

        if not chunks:
            errors.append("Dataset contains 0 chunks.")

        # Check for invalid pages
        empty_pages = 0
        duplicate_hashes = set()
        seen_hashes = set()

        for p in pages:
            if not p.get("title"):
                warnings.append(f"Page missing title: {p.get('url')}")
            if not p.get("content"):
                empty_pages += 1

            chash = p.get("content_hash")
            if chash:
                if chash in seen_hashes:
                    duplicate_hashes.add(chash)
                seen_hashes.add(chash)

        if empty_pages > 0 and len(pages) > 0 and (empty_pages / len(pages)) > 0.5:
            errors.append(f"High empty page ratio: {empty_pages}/{len(pages)}")

        is_valid = len(errors) == 0

        report = {
            "is_valid": is_valid,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "empty_pages": empty_pages,
            "duplicate_pages_count": len(duplicate_hashes),
            "errors": errors,
            "warnings": warnings
        }

        return is_valid, report
