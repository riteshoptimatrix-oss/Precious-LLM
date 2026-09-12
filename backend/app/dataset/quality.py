"""
Precious Edu LLM — Quality Filters

Modular quality filter pipeline:
- EmptyFilter
- LengthFilter
- RepetitionFilter
"""

import re
import logging
from typing import Tuple, Optional
from app.dataset.models import DatasetRecord, RecordType

logger = logging.getLogger(__name__)


class QualityFilter:
    """
    Modular filter pipeline evaluating record content quality.
    """

    def __init__(
        self,
        min_characters: int = 2,
        max_characters: int = 10000,
        max_repetition_ratio: float = 0.5
    ):
        self.min_characters = min_characters
        self.max_characters = max_characters
        self.max_repetition_ratio = max_repetition_ratio

        self.filtered_empty_count = 0
        self.filtered_length_count = 0
        self.filtered_repetition_count = 0

    def evaluate(self, record: DatasetRecord) -> Tuple[bool, Optional[str]]:
        """
        Evaluate record against quality filters.

        Returns:
            Tuple of (passed: bool, filter_reason: Optional[str]).
        """
        text_to_check = self._extract_text(record)

        # 1. Empty Check
        if not text_to_check or len(text_to_check.strip()) == 0:
            self.filtered_empty_count += 1
            return False, "quality_empty_content"

        # 2. Length Check
        char_len = len(text_to_check)
        if char_len < self.min_characters:
            self.filtered_length_count += 1
            return False, f"quality_text_too_short_{char_len}_min_{self.min_characters}"

        if char_len > self.max_characters:
            self.filtered_length_count += 1
            return False, f"quality_text_too_long_{char_len}_max_{self.max_characters}"

        # 3. Pathological Repetition Check
        if self._is_pathological_repetition(text_to_check):
            self.filtered_repetition_count += 1
            return False, "quality_pathological_repetition"

        return True, None

    def _extract_text(self, record: DatasetRecord) -> str:
        """Extract full string text representation from record."""
        if record.type == RecordType.PLAIN_TEXT:
            return record.text or ""
        elif record.type == RecordType.CONVERSATION and record.messages:
            return " ".join(m.content for m in record.messages if m.content)
        return ""

    def _is_pathological_repetition(self, text: str) -> bool:
        """
        Detect extreme pathological character or word repetition (e.g. 'aaaaa...' or 'Hello Hello Hello...').
        Conservative to avoid rejecting legitimate language (e.g. 'Very very good').
        """
        if len(text) < 20:
            return False

        # Single character repeated > 15 times
        if re.search(r"(.)\1{15,}", text):
            return True

        # Short word repeated > 10 times consecutively
        words = text.split()
        if len(words) >= 10:
            max_consecutive = 1
            current_consecutive = 1
            for i in range(1, len(words)):
                if words[i].lower() == words[i - 1].lower():
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 1

            if max_consecutive >= 10:
                return True

        return False
