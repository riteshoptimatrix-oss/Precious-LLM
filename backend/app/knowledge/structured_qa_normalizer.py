"""
Precious Edu LLM — Structured Q&A Query Normalizer & Term Extractor

Implements query cleaning, tokenization, stop word filtering, domain keyword weighting,
and deterministic SHA-256 hash generation for structured Q&A records.
"""

import hashlib
import re
import unicodedata
from typing import List, Set, Tuple


class StructuredQANormalizer:
    """
    Normalizer and domain term extractor for structured Q&A.
    """

    STOP_WORDS: Set[str] = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
        "between", "both", "but", "by", "can", "could", "did", "do", "does", "doing",
        "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
        "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more",
        "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
        "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "please",
        "s", "same", "should", "so", "some", "such", "t", "than", "that", "the", "their",
        "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
        "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
        "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
        "you", "your", "yours", "yourself", "yourselves", "tell", "help", "know", "provide",
        "give", "curious", "details", "right", "deal", "let", "explain", "clarify", "glad",
        "asked", "confirm"
    }

    # High-value domain keywords that receive boosted weights during scoring
    DOMAIN_WEIGHTS = {
        "student": 3.0,
        "visa": 3.0,
        "study": 2.5,
        "ielts": 4.0,
        "coaching": 2.5,
        "classes": 2.5,
        "faculty": 2.5,
        "duration": 3.0,
        "timing": 3.0,
        "time": 2.5,
        "service": 2.5,
        "services": 2.5,
        "documents": 3.5,
        "document": 3.5,
        "admission": 2.5,
        "admissions": 2.5,
        "peic": 5.0,
        "fullform": 4.0,
        "expansion": 4.0,
        "abbreviation": 4.0,
        "canada": 3.0,
        "australia": 3.0,
        "uk": 3.0,
        "usa": 3.0,
        "ireland": 3.0,
        "germany": 3.0,
        "immigration": 2.5,
        "spp": 3.0,
        "sds": 3.0,
        "subclass": 3.0,
        "500": 3.0,
        "privacy": 2.0,
        "policy": 2.0,
        "terms": 2.0,
        "conditions": 2.0,
    }

    # Concept synonyms to bridge phrasing variations
    SYNONYM_EXPANSIONS = {
        "student visa": ["student visa", "study visa", "study permit", "student permit", "sds", "tier 4", "subclass 500"],
        "study visa": ["student visa", "study visa", "study permit", "student permit"],
        "classes": ["classes", "coaching", "training"],
        "coaching": ["classes", "coaching", "training"],
        "timing": ["time", "timing", "timings", "duration", "schedule"],
        "duration": ["time", "timing", "timings", "duration", "schedule"],
        "documents": ["documents", "documentation", "transcripts", "requirements", "checklist"],
        "peic": ["peic", "precious education and immigration consultant", "precious education"],
    }

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Applies Unicode NFKC normalization and strips surrounding whitespace."""
        if not text:
            return ""
        norm = unicodedata.normalize("NFKC", str(text)).strip()
        # Normalize various apostrophe and quote characters to standard single quote
        norm = re.sub(r"[‘’`´]", "'", norm)
        norm = re.sub(r'[“”"]', '"', norm)
        return norm

    @classmethod
    def normalize_input(cls, text: str) -> str:
        """
        Normalizes input question:
        - clean Unicode
        - lowercase
        - strip punctuation (except alphanumeric and single spaces)
        - collapse multiple spaces
        """
        cleaned = cls.clean_text(text).lower()
        # Replace non-alphanumeric (excluding space) with single space
        no_punct = re.sub(r"[^\w\s]", " ", cleaned)
        collapsed = re.sub(r"\s+", " ", no_punct).strip()
        return collapsed

    @classmethod
    def extract_tokens(cls, text: str) -> List[str]:
        """Extracts all non-empty tokens from text."""
        norm = cls.normalize_input(text)
        if not norm:
            return []
        return [tok for tok in norm.split() if tok]

    @classmethod
    def extract_keywords(cls, text: str) -> List[str]:
        """
        Extracts meaningful domain keywords from text:
        - filters stop words
        - keeps tokens with length >= 2
        - deduplicates while preserving order
        """
        tokens = cls.extract_tokens(text)
        seen = set()
        keywords = []
        for t in tokens:
            if t not in cls.STOP_WORDS and len(t) >= 2 and not t.isdigit():
                if t not in seen:
                    seen.add(t)
                    keywords.append(t)
        # If all tokens were filtered (e.g. "what is it"), retain tokens
        if not keywords and tokens:
            for t in tokens:
                if len(t) >= 2 and t not in seen:
                    seen.add(t)
                    keywords.append(t)
        return keywords

    @classmethod
    def generate_record_hash(cls, intent: str, user_input: str, response: str) -> str:
        """
        Generates a deterministic SHA-256 hash for idempotent upserts.
        """
        clean_intent = cls.clean_text(intent).lower()
        clean_input = cls.clean_text(user_input).lower()
        clean_response = cls.clean_text(response)
        raw_key = f"{clean_intent}::{clean_input}::{clean_response}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @classmethod
    def detect_country(cls, text: str) -> Optional[str]:
        """Detects canonical country entity from text."""
        if not text:
            return None
        cleaned = text.lower()
        if re.search(r"\b(usa|united states|america)\b", cleaned):
            return "USA"
        if re.search(r"\b(canada|canadian)\b", cleaned):
            return "Canada"
        if re.search(r"\b(australia|australian)\b", cleaned):
            return "Australia"
        if re.search(r"\b(uk|united kingdom|britain|british|england)\b", cleaned):
            return "UK"
        if re.search(r"\b(new zealand|nz)\b", cleaned):
            return "New Zealand"
        if re.search(r"\b(singapore)\b", cleaned):
            return "Singapore"
        if re.search(r"\b(ireland|irish)\b", cleaned):
            return "Ireland"
        if re.search(r"\b(malaysia)\b", cleaned):
            return "Malaysia"
        if re.search(r"\b(in us|to us|for us|study in us|us visa|us student|us tourist|us b1|us b2)\b", cleaned):
            return "USA"
        return None

    @classmethod
    def detect_visa_type(cls, text: str) -> Optional[str]:
        """Detects canonical visa/service category from text."""
        if not text:
            return None
        cleaned = text.lower()
        if any(w in cleaned for w in ["visitor", "tourist", "visit visa", "super visa", "trv", "short stay", "b1/b2", "b1b2"]):
            return "visitor"
        if any(w in cleaned for w in ["student visa", "study visa", "study permit", "student permit", "admission", "study", "university", "college", "f1", "m1", "subclass 500", "sds"]):
            return "student"
        if any(w in cleaned for w in ["work permit", "work visa", "lmia", "pgwp", "sowp", "open work", "worker"]):
            return "work"
        if any(w in cleaned for w in ["express entry", "pnp", "oinp", "rcip", "aipp", "pr card", "citizenship", "permanent residency", "pr"]):
            return "immigration"
        if any(w in cleaned for w in ["ielts", "toefl", "pte"]):
            return "ielts"
        if any(w in cleaned for w in ["email", "phone", "mobile", "contact", "address", "call", "number"]):
            return "contact"
        return None

    @classmethod
    def clean_boilerplate(cls, text: str) -> str:
        """Strips repetitive template prefixes and trailing promotional URLs."""
        if not text:
            return ""
        prefix_pattern = re.compile(
            r'^(?:okay,\s*here\s+is\s+the\s+information:\s*|'
            r'here\'s\s+what\s+you\s+need\s+to\s+know:\s*|'
            r'here\s+is\s+the\s+information:\s*|'
            r'our\s+team\s+says:\s*|'
            r'to\s+answer\s+your\s+question:\s*|'
            r'we\s+can\s+confirm\s+that\s*|'
            r'according\s+to\s+our\s+details,\s*|'
            r'good\s+question[.!:]?\s*|'
            r'glad\s+you\s+asked[.!:]?\s*|'
            r'absolutely[.!:]?\s*|'
            r'of\s+course[.!:]?\s*|'
            r'well,\s*|'
            r'as\s+per\s+our\s+services,\s*|'
            r'let\s+me\s+clarify:\s*|'
            r'i\'d\s+be\s+happy\s+to\s+help[\.!]?\s*)+',
            re.IGNORECASE
        )
        cleaned = prefix_pattern.sub('', text).strip()
        suffix_pattern = re.compile(
            r'(?:\s*(?:for\s+(?:more\s+)?details,?\s*visit:?|visit:?|check\s+out\s+our\s+website:?|check\s+our\s+services(?:\s+here)?:?|discover\s+more(?:\s+at)?:?|read\s+more(?:\s+at)?:?|learn\s+more(?:\s+at)?:?|more\s+info:?)\s*(?:https?://\S+)?|\s*https?://\S+)+$',
            re.IGNORECASE
        )
        cleaned = suffix_pattern.sub('', cleaned).strip()
        if cleaned.endswith((':', ',')):
            cleaned = cleaned[:-1].rstrip() + '.'
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned or text



