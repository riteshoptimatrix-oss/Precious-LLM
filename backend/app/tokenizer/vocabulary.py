"""
Precious Edu LLM — Vocabulary Manager

Manages bidirectional mapping between token strings and integer token IDs.
Enforces vocabulary integrity: uniqueness, contiguity, and stable special token IDs.
"""

import logging
from typing import Dict, List, Optional, Set
from app.tokenizer.exceptions import InvalidVocabularyError

logger = logging.getLogger(__name__)


class Vocabulary:
    """
    Bidirectional token vocabulary mapping (token <-> ID).
    """

    def __init__(self, token_to_id: Optional[Dict[str, int]] = None):
        self.token_to_id: Dict[str, int] = token_to_id.copy() if token_to_id else {}
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.token_to_id.items()}

    def add_token(self, token: str, forced_id: Optional[int] = None) -> int:
        """
        Add a token string to the vocabulary and return its integer ID.
        """
        if token in self.token_to_id:
            return self.token_to_id[token]

        if forced_id is not None:
            token_id = forced_id
        else:
            token_id = len(self.token_to_id)

        if token_id in self.id_to_token and self.id_to_token[token_id] != token:
            raise InvalidVocabularyError(
                f"Token ID collision: ID {token_id} already assigned to '{self.id_to_token[token_id]}', cannot assign to '{token}'"
            )

        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        return token_id

    def get_id(self, token: str) -> Optional[int]:
        """Get ID for a token string, or None if not found."""
        return self.token_to_id.get(token)

    def get_token(self, token_id: int) -> Optional[str]:
        """Get token string for an integer ID, or None if not found."""
        return self.id_to_token.get(token_id)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

    def validate_integrity(self) -> bool:
        """
        Validate vocabulary constraints:
        - Total tokens match length of mappings.
        - IDs are contiguous integers from 0 to len(vocab) - 1.
        - No duplicate tokens or IDs exist.
        """
        vocab_size = len(self.token_to_id)
        if vocab_size != len(self.id_to_token):
            raise InvalidVocabularyError(
                f"Vocabulary size mismatch: token_to_id ({vocab_size}) != id_to_token ({len(self.id_to_token)})"
            )

        for expected_id in range(vocab_size):
            if expected_id not in self.id_to_token:
                raise InvalidVocabularyError(f"Vocabulary ID gap detected: Missing expected contiguous token ID {expected_id}")
            token = self.id_to_token[expected_id]
            if self.token_to_id.get(token) != expected_id:
                raise InvalidVocabularyError(f"Vocabulary mapping inconsistency for token '{token}' at ID {expected_id}")

        return True
