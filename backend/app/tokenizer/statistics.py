"""
Precious Edu LLM — Tokenizer Statistics Calculator

Calculates empirical statistics across tokenized sequences:
- Raw character count vs token count
- Compression ratio (characters per token)
- Unknown token rate (<unk> count / total tokens)
- Token sequence length percentiles (P50, P75, P90, P95, P99, Max) for Phase 6 Transformer context sizing.
"""

import math
import logging
from typing import Dict, Any, List, Optional
from app.tokenizer.config import TokenizerConfig, get_tokenizer_config

logger = logging.getLogger(__name__)


class TokenizerStatisticsCalculator:
    """
    Calculates empirical tokenization metrics across sequences.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or get_tokenizer_config()

    def calculate_statistics(
        self,
        raw_texts: List[str],
        token_id_sequences: List[List[int]],
        vocab_size: int,
        merges_count: int
    ) -> Dict[str, Any]:
        """
        Calculate metrics for raw texts and encoded token ID sequences.
        """
        num_records = len(token_id_sequences)
        if num_records == 0:
            return {}

        unk_id = self.config.special_tokens_map[self.config.UNK_TOKEN]
        total_raw_chars = sum(len(txt) for txt in raw_texts)
        total_tokens = sum(len(seq) for seq in token_id_sequences)
        total_unk_tokens = sum(seq.count(unk_id) for seq in token_id_sequences)

        token_lengths = [len(seq) for seq in token_id_sequences]
        token_lengths.sort()

        def get_percentile(p: float) -> int:
            if not token_lengths:
                return 0
            k = (len(token_lengths) - 1) * (p / 100.0)
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return token_lengths[int(k)]
            return int(round(token_lengths[f] * (c - k) + token_lengths[c] * (k - f)))

        compression_ratio = round(total_raw_chars / total_tokens, 3) if total_tokens > 0 else 0.0
        unk_rate_pct = round((total_unk_tokens / total_tokens * 100), 4) if total_tokens > 0 else 0.0
        avg_tokens_per_record = round(total_tokens / num_records, 2) if num_records > 0 else 0.0

        return {
            "training_records_count": num_records,
            "vocab_size": vocab_size,
            "merges_count": merges_count,
            "total_raw_characters": total_raw_chars,
            "total_tokens": total_tokens,
            "total_unknown_tokens": total_unk_tokens,
            "unknown_token_rate_pct": unk_rate_pct,
            "compression_ratio_chars_per_token": compression_ratio,
            "avg_tokens_per_record": avg_tokens_per_record,
            "sequence_length_percentiles": {
                "p50": get_percentile(50),
                "p75": get_percentile(75),
                "p90": get_percentile(90),
                "p95": get_percentile(95),
                "p99": get_percentile(99),
                "min": min(token_lengths) if token_lengths else 0,
                "max": max(token_lengths) if token_lengths else 0,
            }
        }
