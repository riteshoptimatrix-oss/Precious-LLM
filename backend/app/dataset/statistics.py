"""
Precious Edu LLM — Dataset Statistics Calculator

Calculates comprehensive empirical statistics across processed records:
- Total record counts & type distributions
- Conversation, turn, and role counts (user, assistant, system)
- Character length statistics (min, max, average, total)
- Language distribution
- Source provenance distribution
"""

import logging
from typing import Dict, Any, List, Optional
from app.dataset.models import DatasetRecord, RecordType

logger = logging.getLogger(__name__)


class DatasetStatisticsCalculator:
    """
    Calculates empirical statistics for a dataset collection.
    """

    def calculate_statistics(self, records: List[DatasetRecord]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of DatasetRecord objects.
        """
        total_records = len(records)
        if total_records == 0:
            return {
                "total_records": 0,
                "total_conversations": 0,
                "total_plain_texts": 0,
                "total_messages": 0,
                "role_distribution": {},
                "character_stats": {
                    "min_characters": 0,
                    "max_characters": 0,
                    "avg_characters": 0.0,
                    "total_characters": 0
                },
                "language_distribution": {},
                "source_distribution": {}
            }

        conversations_count = 0
        plain_texts_count = 0
        total_messages = 0
        role_dist: Dict[str, int] = {}
        language_dist: Dict[str, int] = {}
        source_dist: Dict[str, int] = {}
        char_lengths: List[int] = []

        for rec in records:
            # Language distribution
            lang = rec.language or "en"
            language_dist[lang] = language_dist.get(lang, 0) + 1

            # Source distribution
            src = rec.source_id or "unknown"
            source_dist[src] = source_dist.get(src, 0) + 1

            if rec.type == RecordType.PLAIN_TEXT:
                plain_texts_count += 1
                length = len(rec.text or "")
                char_lengths.append(length)

            elif rec.type == RecordType.CONVERSATION and rec.messages:
                conversations_count += 1
                conv_len = 0
                for msg in rec.messages:
                    total_messages += 1
                    role = msg.role.lower()
                    role_dist[role] = role_dist.get(role, 0) + 1
                    conv_len += len(msg.content or "")
                char_lengths.append(conv_len)

        total_chars = sum(char_lengths)
        min_chars = min(char_lengths) if char_lengths else 0
        max_chars = max(char_lengths) if char_lengths else 0
        avg_chars = round(total_chars / len(char_lengths), 2) if char_lengths else 0.0

        return {
            "total_records": total_records,
            "total_conversations": conversations_count,
            "total_plain_texts": plain_texts_count,
            "total_messages": total_messages,
            "role_distribution": role_dist,
            "character_stats": {
                "min_characters": min_chars,
                "max_characters": max_chars,
                "avg_characters": avg_chars,
                "total_characters": total_chars
            },
            "language_distribution": language_dist,
            "source_distribution": source_dist
        }
