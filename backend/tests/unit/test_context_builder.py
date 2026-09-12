"""
Precious Edu LLM — Context Builder Unit Tests
"""

import pytest
from app.conversation.context_builder import ContextBuilder, ConversationContext


def test_context_builder_builds_valid_context():
    cb = ContextBuilder(system_instructions="Custom Instructions")
    context = cb.build_context(
        current_user_message="What is my name?",
        recent_messages=[
            {"role": "user", "content": "My name is Ritesh."},
            {"role": "assistant", "content": "Nice to meet you, Ritesh."}
        ],
        memories={"user_name": "Ritesh"},
        metadata={"turn": 3}
    )

    assert isinstance(context, ConversationContext)
    assert context.system_instructions == "Custom Instructions"
    assert context.current_user_message == "What is my name?"
    assert context.memories == {"user_name": "Ritesh"}
    assert len(context.recent_messages) == 2
    assert context.metadata == {"turn": 3}


def test_context_builder_defaults():
    cb = ContextBuilder()
    context = cb.build_context(current_user_message="Hello")

    assert context.system_instructions is not None
    assert context.memories == {}
    assert context.recent_messages == []
    assert context.current_user_message == "Hello"
