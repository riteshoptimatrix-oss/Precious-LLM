"""
Precious Edu LLM — Text Normalizer

Normalizes raw text before tokenization:
- Unicode NFKC normalization
- Whitespace normalization
- Optional lowercasing
- Control character removal
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)


class TextNormalizer:
    """
    Text normalization pipeline for the tokenizer.

    Applied to all text before BPE tokenization.
    """

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase

    def normalize(self, text: str) -> str:
        """
        Apply the full normalization pipeline to input text.

        Args:
            text: Raw input text.

        Returns:
            Normalized text.
        """
        if not text:
            return text

        # Unicode NFKC normalization (canonical decomposition + compatibility composition)
        text = unicodedata.normalize("NFKC", text)

        # Remove control characters (keep newlines and tabs)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalize whitespace: collapse multiple spaces/tabs into single space
        text = re.sub(r"[ \t]+", " ", text)

        # Normalize newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Optional lowercase
        if self.lowercase:
            text = text.lower()

        return text
