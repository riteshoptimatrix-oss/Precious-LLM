"""
Precious Edu LLM — BPE Decoder

Decodes integer token IDs back into human-readable text string.
Supports optional special token filtering.
"""

import logging
from typing import List, Optional
from app.tokenizer.config import TokenizerConfig, get_tokenizer_config
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.exceptions import UnknownTokenError

logger = logging.getLogger(__name__)


class BPEDecoder:
    """
    Decodes integer token ID sequences into strings.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        config: Optional[TokenizerConfig] = None
    ):
        self.vocab = vocab
        self.config = config or get_tokenizer_config()

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        Decode a list of integer token IDs into a reconstructed string.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: If True, strips special token markers from output.

        Returns:
            Reconstructed string.
        """
        if not token_ids:
            return ""

        special_set = set(self.config.special_tokens_list)
        text_parts: List[str] = []

        for tid in token_ids:
            token = self.vocab.get_token(tid)
            if token is None:
                token = self.config.UNK_TOKEN

            if skip_special_tokens and token in special_set:
                continue

            text_parts.append(token)

        return "".join(text_parts)
