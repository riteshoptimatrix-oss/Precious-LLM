"""
Precious Edu LLM — Conversation Formatter

Formats Phase 4 dataset records into tokenizable text sequences with configured special role markers.
"""

from typing import Any, Dict, List, Optional
from app.dataset.models import DatasetRecord, RecordType
from app.tokenizer.config import TokenizerConfig, get_tokenizer_config


class ConversationFormatter:
    """
    Formats conversation records into structured tokenizable string streams.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or get_tokenizer_config()

    def format_record(self, record: DatasetRecord) -> str:
        """
        Format a single DatasetRecord into a tokenizable string sequence.
        """
        if record.type == RecordType.PLAIN_TEXT and record.text:
            return record.text

        elif record.type == RecordType.CONVERSATION and record.messages:
            parts = [self.config.BOS_TOKEN]
            for msg in record.messages:
                role = msg.role.lower().strip()
                if role == "user":
                    role_token = self.config.USER_TOKEN
                elif role == "assistant":
                    role_token = self.config.ASSISTANT_TOKEN
                elif role == "system":
                    role_token = self.config.SYSTEM_TOKEN
                else:
                    role_token = f"<{role}>"

                parts.append(f"{role_token}\n{msg.content.strip()}")

            parts.append(self.config.EOS_TOKEN)
            return "\n".join(parts)

        return ""
