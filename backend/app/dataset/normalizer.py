"""
Precious Edu LLM — Dataset Normalizer

Executes deterministic, conservative text normalization:
- Normalizes line endings (\r\n -> \n)
- Removes null bytes (\x00)
- Trims leading/trailing whitespace and collapses 3+ consecutive blank lines into 2
- Preserves natural punctuation, capitalization, symbols, accents, emojis, and full Unicode scripts.
"""

import re
import logging
from app.dataset.models import DatasetRecord, RecordType

logger = logging.getLogger(__name__)


class DatasetNormalizer:
    """
    Normalizes dataset text while preserving natural linguistic information.
    """

    def normalize_text(self, text: str) -> str:
        """
        Normalize a raw text string conservatively.
        """
        if not text:
            return ""

        # 1. Remove null bytes
        text = text.replace("\x00", "")

        # 2. Normalize line endings (\r\n -> \n, \r -> \n)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Normalize horizontal whitespace (tab/multiple space -> single space per line)
        lines = text.split("\n")
        cleaned_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in lines]

        # 4. Collapse 3+ consecutive blank lines into max 2
        result_lines = []
        blank_count = 0
        for line in cleaned_lines:
            if not line:
                blank_count += 1
                if blank_count <= 1:
                    result_lines.append(line)
            else:
                blank_count = 0
                result_lines.append(line)

        # 5. Trim overall string
        return "\n".join(result_lines).strip()

    def normalize_record(self, record: DatasetRecord) -> DatasetRecord:
        """
        Normalize a canonical DatasetRecord in place.
        """
        if record.type == RecordType.PLAIN_TEXT and record.text:
            record.text = self.normalize_text(record.text)

        elif record.type == RecordType.CONVERSATION and record.messages:
            for msg in record.messages:
                msg.content = self.normalize_text(msg.content)

        return record
