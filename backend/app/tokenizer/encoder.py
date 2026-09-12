"""
Precious Edu LLM — BPE Encoder

Encodes raw text into integer token IDs using learned BPE merge rules.
Handles special tokens, preserves whitespace, and maps unknown characters to <unk>.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from app.tokenizer.config import TokenizerConfig, get_tokenizer_config
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.bpe import BPEAlgorithm

logger = logging.getLogger(__name__)


class BPEEncoder:
    """
    Encodes text strings into integer token IDs.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        merges: List[Tuple[str, str]],
        config: Optional[TokenizerConfig] = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.config = config or get_tokenizer_config()
        self.bpe = BPEAlgorithm(special_tokens=set(self.config.special_tokens_list))

        # Precompute pair-to-rank map for fast lookup during encoding
        self.merge_ranks: Dict[Tuple[str, str], int] = {pair: rank for rank, pair in enumerate(self.merges)}

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text into integer token IDs.

        Args:
            text: Raw input string.
            add_bos: If True, prepends BOS token ID.
            add_eos: If True, appends EOS token ID.

        Returns:
            List of integer token IDs.
        """
        if not text:
            token_ids: List[int] = []
            if add_bos:
                token_ids.insert(0, self.config.special_tokens_map[self.config.BOS_TOKEN])
            if add_eos:
                token_ids.append(self.config.special_tokens_map[self.config.EOS_TOKEN])
            return token_ids

        # 1. Tokenize text into initial symbols
        symbols = self.bpe.tokenize_to_symbols(text)

        # 2. Apply BPE merges iteratively according to learned merge ranks
        symbols = self._apply_merges(symbols)

        # 3. Map symbols to Vocabulary integer IDs
        token_ids: List[int] = []
        unk_id = self.config.special_tokens_map[self.config.UNK_TOKEN]

        for sym in symbols:
            tid = self.vocab.get_id(sym)
            if tid is not None:
                token_ids.append(tid)
            else:
                # Symbol unknown: map to <unk>
                token_ids.append(unk_id)

        # 4. Optional BOS / EOS handling
        if add_bos:
            token_ids.insert(0, self.config.special_tokens_map[self.config.BOS_TOKEN])
        if add_eos:
            token_ids.append(self.config.special_tokens_map[self.config.EOS_TOKEN])

        return token_ids

    def _apply_merges(self, symbols: List[str]) -> List[str]:
        """
        Iteratively apply learned merges to symbol list.
        """
        if len(symbols) < 2 or not self.merge_ranks:
            return symbols

        while len(symbols) >= 2:
            # Find all available adjacent pairs in current symbols
            pairs = []
            for i in range(len(symbols) - 1):
                sym1, sym2 = symbols[i], symbols[i + 1]
                if sym1 not in self.config.special_tokens_list and sym2 not in self.config.special_tokens_list:
                    pair = (sym1, sym2)
                    if pair in self.merge_ranks:
                        pairs.append((self.merge_ranks[pair], pair, i))

            if not pairs:
                break

            # Pick pair with lowest rank (first learned merge)
            pairs.sort(key=lambda x: x[0])
            best_rank, best_pair, _ = pairs[0]

            new_token = best_pair[0] + best_pair[1]
            symbols = self.bpe.merge_pair_in_sequences([symbols], best_pair, new_token)[0]

        return symbols
