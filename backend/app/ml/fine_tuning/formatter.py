"""
Precious Edu LLM — Conversation Formatter for Fine-Tuning

Formats canonical conversation messages into tokenizable text sequences with special role tokens.
"""

from typing import Dict, List, Optional
from app.tokenizer.config import TokenizerConfig, get_tokenizer_config


class ConversationFormatter:
    """
    Formats structured message lists into prompt sequences for tokenization and fine-tuning.
    """

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or get_tokenizer_config()

    def format_messages(self, messages: List[Dict[str, str]], add_eos: bool = True) -> str:
        """
        Formats a list of message dicts: [{'role': 'user', 'content': 'Hello'}]
        into standard text sequence format:
        <system>\nContent\n<user>\nContent\n<assistant>\nContent\n<eos>
        """
        parts = []
        for msg in messages:
            role = msg["role"].lower().strip()
            content = msg["content"].strip()

            if role == "system":
                token = self.config.SYSTEM_TOKEN
            elif role == "user":
                token = self.config.USER_TOKEN
            elif role == "assistant":
                token = self.config.ASSISTANT_TOKEN
            else:
                token = f"<{role}>"

            parts.append(f"{token}\n{content}")

        text = "\n".join(parts)
        if add_eos:
            text += f"\n{self.config.EOS_TOKEN}"
        return text

    def format_prompt_only(self, messages: List[Dict[str, str]]) -> str:
        """
        Formats conversation history up to the latest user message, ending with
        '<assistant>\n' to prompt assistant generation.
        """
        parts = []
        for msg in messages:
            role = msg["role"].lower().strip()
            content = msg["content"].strip()

            if role == "assistant" and msg is messages[-1]:
                # If last turn in passed array is assistant, skip it to generate
                continue

            if role == "system":
                token = self.config.SYSTEM_TOKEN
            elif role == "user":
                token = self.config.USER_TOKEN
            elif role == "assistant":
                token = self.config.ASSISTANT_TOKEN
            else:
                token = f"<{role}>"

            parts.append(f"{token}\n{content}")

        parts.append(f"{self.config.ASSISTANT_TOKEN}\n")
        return "\n".join(parts)
