"""
Precious Edu LLM — BPE Algorithm Unit Tests
"""

import pytest
from app.tokenizer.bpe import BPEAlgorithm
from app.tokenizer.vocabulary import Vocabulary


def test_bpe_pair_counting():
    bpe = BPEAlgorithm()
    seqs = [["l", "o", "o", "k"], ["l", "o", "o", "k", "i", "n", "g"]]
    counts = bpe.get_pair_frequencies(seqs)

    assert counts[("l", "o")] == 2
    assert counts[("o", "o")] == 2
    assert counts[("o", "k")] == 2


def test_bpe_deterministic_tie_breaking():
    bpe = BPEAlgorithm()
    # Create scenario where pair (a, b) and (c, d) both have frequency 3
    seqs = [["a", "b"], ["a", "b"], ["a", "b"], ["c", "d"], ["c", "d"], ["c", "d"]]
    counts = bpe.get_pair_frequencies(seqs)

    # Deterministic tie-breaker should pick ('a', 'b') first because 'a' < 'c' lexically
    best = bpe.get_best_pair(counts)
    assert best == ("a", "b")


def test_bpe_merge_sequences():
    bpe = BPEAlgorithm()
    seqs = [["h", "e", "l", "l", "o"]]
    merged = bpe.merge_pair_in_sequences(seqs, ("l", "l"), "ll")

    assert merged == [["h", "e", "ll", "o"]]
