"""
Unit tests for ConversationFormatter in app.ml.fine_tuning.formatter
"""

import pytest
from app.ml.fine_tuning.formatter import ConversationFormatter


def test_format_messages():
    formatter = ConversationFormatter()
    messages = [
        {"role": "system", "content": "You are Precious AI."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"}
    ]
    formatted = formatter.format_messages(messages, add_eos=True)

    assert "<system>" in formatted
    assert "You are Precious AI." in formatted
    assert "<user>" in formatted
    assert "Hello" in formatted
    assert "<assistant>" in formatted
    assert "Hi! How can I help?" in formatted
    assert "<eos>" in formatted


def test_format_prompt_only():
    formatter = ConversationFormatter()
    messages = [
        {"role": "user", "content": "Hello"}
    ]
    prompt_str = formatter.format_prompt_only(messages)

    assert "<user>\nHello" in prompt_str
    assert prompt_str.endswith("<assistant>\n")
