"""
Precious Edu LLM — Conversational Data Validator

Validates conversational data records prior to tokenization and fine-tuning.
Enforces structural integrity, valid role turn sequences, non-empty messages,
Unicode sanity, and duplicate prevention.
"""

import logging
from typing import Any, Dict, List, Tuple
from app.ml.fine_tuning.exceptions import DatasetValidationError

logger = logging.getLogger(__name__)

VALID_ROLES = {"system", "user", "assistant"}


class ConversationValidator:
    """
    Validates conversation payloads before tokenization.
    """

    def __init__(self, allow_consecutive_user: bool = False, max_turns: int = 50):
        self.allow_consecutive_user = allow_consecutive_user
        self.max_turns = max_turns

    def normalize_record(self, raw_record: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Normalizes single-turn, multi-turn, or instruction-response inputs into
        a canonical list of message objects: [{'role': '...', 'content': '...'}]
        """
        if not isinstance(raw_record, dict):
            raise DatasetValidationError("Record must be a JSON dictionary.")

        # Case 1: Standard 'messages' array format
        if "messages" in raw_record:
            messages = raw_record["messages"]
            if not isinstance(messages, list):
                raise DatasetValidationError("'messages' field must be a list.")
            return messages

        # Case 2: 'instruction' and 'response' format
        if "instruction" in raw_record and "response" in raw_record:
            system_prompt = raw_record.get("system", None)
            msgs = []
            if system_prompt and isinstance(system_prompt, str) and system_prompt.strip():
                msgs.append({"role": "system", "content": system_prompt.strip()})
            
            instruction = str(raw_record["instruction"]).strip()
            response = str(raw_record["response"]).strip()
            
            msgs.append({"role": "user", "content": instruction})
            msgs.append({"role": "assistant", "content": response})
            return msgs

        raise DatasetValidationError("Record must contain either 'messages' list or 'instruction'/'response' keys.")

    def validate_conversation(self, raw_record: Dict[str, Any]) -> Tuple[bool, str, List[Dict[str, str]]]:
        """
        Validate single conversation record.
        Returns:
            (is_valid, error_reason, normalized_messages)
        """
        try:
            messages = self.normalize_record(raw_record)
        except DatasetValidationError as e:
            return False, str(e), []

        if not messages:
            return False, "Conversation contains no messages.", []

        if len(messages) > self.max_turns:
            return False, f"Conversation turn count ({len(messages)}) exceeds max_turns ({self.max_turns}).", []

        prev_role = None
        has_assistant_response = False

        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, f"Message at index {idx} is not an object.", []

            role = msg.get("role", "").lower().strip()
            content = msg.get("content", "")

            if role not in VALID_ROLES:
                return False, f"Invalid role '{role}' at index {idx}. Valid roles: {VALID_ROLES}", []

            if not isinstance(content, str) or not content.strip():
                return False, f"Empty or non-string content for role '{role}' at index {idx}.", []

            # Sanity check Unicode null bytes or corruption
            if "\x00" in content:
                return False, f"Null byte detected in content at index {idx}.", []

            # System message must be first turn if present
            if role == "system" and idx != 0:
                return False, f"System role must appear only as the first message (index 0), got index {idx}.", []

            # Role order check
            if prev_role:
                if role == "assistant" and prev_role == "assistant":
                    return False, f"Consecutive assistant turns detected at index {idx}.", []
                if role == "user" and prev_role == "user" and not self.allow_consecutive_user:
                    return False, f"Consecutive user turns detected at index {idx}.", []

            if role == "assistant":
                has_assistant_response = True

            prev_role = role

        if not has_assistant_response:
            return False, "Conversation lacks required assistant response for fine-tuning.", []

        return True, "", messages
