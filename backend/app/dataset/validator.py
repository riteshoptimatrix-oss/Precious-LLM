"""
Precious Edu LLM — Dataset Validator

Validates DatasetRecords for structural correctness, role validity, non-empty content,
and message sequence rules.
"""

import logging
from typing import List, Tuple, Optional
from app.dataset.models import DatasetRecord, RecordType

logger = logging.getLogger(__name__)

ALLOWED_ROLES = {"user", "assistant", "system"}


class DatasetValidator:
    """
    Validates DatasetRecords before inclusion in training splits.
    """

    def validate_record(self, record: DatasetRecord) -> Tuple[bool, Optional[str]]:
        """
        Validate a DatasetRecord instance.

        Returns:
            Tuple of (is_valid: bool, rejection_reason: Optional[str]).
        """
        # Check raw parse error metadata
        if "raw_parse_error" in record.metadata:
            return False, f"malformed_json: {record.metadata['raw_parse_error']}"

        if record.type == RecordType.PLAIN_TEXT:
            return self._validate_plain_text(record)

        elif record.type == RecordType.CONVERSATION:
            return self._validate_conversation(record)

        return False, f"unsupported_record_type_{record.type}"

    def _validate_plain_text(self, record: DatasetRecord) -> Tuple[bool, Optional[str]]:
        """Validate plain text record."""
        if not record.text or not record.text.strip():
            return False, "empty_content"
        return True, None

    def _validate_conversation(self, record: DatasetRecord) -> Tuple[bool, Optional[str]]:
        """Validate conversation record."""
        if not record.messages or len(record.messages) == 0:
            return False, "empty_conversation_messages"

        # Check each message
        for idx, msg in enumerate(record.messages):
            role = msg.role.lower().strip() if msg.role else ""
            if role not in ALLOWED_ROLES:
                return False, f"invalid_role_{msg.role}_at_index_{idx}"

            if not msg.content or not msg.content.strip():
                return False, f"empty_message_content_at_index_{idx}"

        # Check turn sequence (at least 1 message, optional system start, valid roles)
        has_meaningful_turn = any(m.role in ("user", "assistant") for m in record.messages)
        if not has_meaningful_turn:
            return False, "no_user_or_assistant_turns"

        return True, None
