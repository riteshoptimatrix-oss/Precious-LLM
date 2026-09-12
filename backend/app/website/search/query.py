"""
Precious AI — Website Query Normalizer & Terminology Expander
"""

import re
from typing import List, Set


class WebsiteQueryNormalizer:
    """
    Normalizes search query terms, filters common stop words, and expands domain synonyms.
    """

    STOP_WORDS = {
        "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did",
        "a", "an", "the", "and", "but", "if", "or", "because",
        "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up",
        "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where",
        "why", "how", "all", "any", "both", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "can", "will",
        "just", "don", "should", "now", "what", "which", "who", "whom",
        "me", "my", "i", "you", "your", "help"
    }

    SYNONYM_MAP = {
        "f1": ["f1", "f-1", "student visa", "academic visa"],
        "m1": ["m1", "m-1", "vocational visa"],
        "coaching": ["coaching", "training", "classes", "ielts", "toefl", "pte", "gre", "gmat"],
        "country": ["country", "countries", "usa", "uk", "canada", "australia", "germany", "singapore", "malaysia", "ireland"],
        "countries": ["country", "countries", "usa", "uk", "canada", "australia", "germany", "singapore", "malaysia", "ireland"],
        "service": ["service", "services", "assistance", "support", "guidance"],
        "services": ["service", "services", "assistance", "support", "guidance"],
        "visa": ["visa", "permit", "immigration", "pr", "admission", "student visa", "study visa"],
        "abroad": ["abroad", "overseas", "foreign", "countries"],
    }

    @classmethod
    def normalize_query(cls, query_text: str) -> List[str]:
        """
        Tokenizes, normalizes, filters stop words, and expands query terms.
        """
        if not query_text:
            return []
        cleaned = re.sub(r'[^\w\s-]', ' ', query_text.lower())
        raw_tokens = [t.strip() for t in cleaned.split() if len(t.strip()) > 1]

        # Filter out common stop words
        filtered_tokens = [t for t in raw_tokens if t not in cls.STOP_WORDS]
        if not filtered_tokens and raw_tokens:
            filtered_tokens = raw_tokens

        expanded = set(filtered_tokens)
        for tok in filtered_tokens:
            if tok in cls.SYNONYM_MAP:
                expanded.update(cls.SYNONYM_MAP[tok])

        return list(expanded)
