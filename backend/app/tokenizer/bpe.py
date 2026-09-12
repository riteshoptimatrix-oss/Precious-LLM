"""
Precious Edu LLM — Pure Python BPE Algorithm

Implements character/symbol-level Byte-Pair Encoding:
- Special token protection (special tokens like <user> are preserved intact)
- Adjacent pair frequency counting
- Deterministic tie-breaking (frequency DESC, then pair tuple ASC)
- Iterative symbol pair merging
"""

import re
import logging
from collections import Counter
from typing import Dict, List, Set, Tuple, Optional

logger = logging.getLogger(__name__)


class BPEAlgorithm:
    """
    Pure Python BPE algorithm.
    """

    def __init__(self, special_tokens: Optional[Set[str]] = None):
        self.special_tokens: Set[str] = special_tokens or set()
        # Compile regex pattern to match special tokens or individual characters
        if self.special_tokens:
            # Sort special tokens by length DESC so longer special tokens match first
            escaped = [re.escape(st) for st in sorted(self.special_tokens, key=len, reverse=True)]
            self.split_pattern = re.compile("|".join(escaped) + r"|.")
        else:
            self.split_pattern = None

    def tokenize_to_symbols(self, text: str) -> List[str]:
        """
        Split raw text into initial symbol sequence.
        Special tokens remain intact as single symbol strings; ordinary text is split into characters.
        """
        if not text:
            return []

        if self.split_pattern:
            return self.split_pattern.findall(text)
        else:
            return list(text)

    def get_pair_frequencies(self, sequences: List[List[str]]) -> Counter:
        """
        Count adjacent symbol pairs across all sequences.
        Pairs containing special tokens are excluded from merging.
        """
        counts: Counter = Counter()
        for seq in sequences:
            for i in range(len(seq) - 1):
                sym1, sym2 = seq[i], seq[i + 1]
                # Skip pair if either symbol is a special token
                if sym1 in self.special_tokens or sym2 in self.special_tokens:
                    continue
                counts[(sym1, sym2)] += 1
        return counts

    def get_best_pair(self, pair_counts: Counter) -> Optional[Tuple[str, str]]:
        """
        Select best pair using deterministic tie-breaking:
        1. Primary sort key: frequency DESC (-count)
        2. Secondary sort key: lexical pair tuple ASC ((sym1, sym2))
        """
        if not pair_counts:
            return None

        # Sort pairs deterministically
        sorted_pairs = sorted(
            pair_counts.items(),
            key=lambda item: (-item[1], item[0])
        )

        return sorted_pairs[0][0]

    def merge_pair_in_sequences(
        self,
        sequences: List[List[str]],
        pair: Tuple[str, str],
        new_token: str
    ) -> List[List[str]]:
        """
        Replace all adjacent occurrences of pair = (sym1, sym2) with new_token in sequences.
        """
        sym1, sym2 = pair
        merged_sequences: List[List[str]] = []

        for seq in sequences:
            new_seq = []
            i = 0
            n = len(seq)
            while i < n:
                if i < n - 1 and seq[i] == sym1 and seq[i + 1] == sym2:
                    new_seq.append(new_token)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            merged_sequences.append(new_seq)

        return merged_sequences
