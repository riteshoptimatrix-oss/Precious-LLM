"""
Unit tests for ConversationValidator in app.ml.fine_tuning.validator
"""

import pytest
from app.ml.fine_tuning.validator import ConversationValidator
from app.ml.fine_tuning.exceptions import DatasetValidationError


def test_validator_valid_multi_turn():
    validator = ConversationValidator()
    record = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "My name is Ritesh."},
            {"role": "assistant", "content": "Nice to meet you Ritesh!"}
        ]
    }
    is_valid, reason, messages = validator.validate_conversation(record)
    assert is_valid is True
    assert reason == ""
    assert len(messages) == 4


def test_validator_instruction_response_format():
    validator = ConversationValidator()
    record = {
        "system": "You are Precious AI.",
        "instruction": "What is an F1 visa?",
        "response": "An F1 visa is a U.S. student visa."
    }
    is_valid, reason, messages = validator.validate_conversation(record)
    assert is_valid is True
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"


def test_validator_rejects_empty_messages():
    validator = ConversationValidator()
    record = {"messages": []}
    is_valid, reason, messages = validator.validate_conversation(record)
    assert is_valid is False
    assert "no messages" in reason.lower()


def test_validator_rejects_consecutive_assistant():
    validator = ConversationValidator()
    record = {
        "messages": [
            {"role": "assistant", "content": "Hi"},
            {"role": "assistant", "content": "Hello again"}
        ]
    }
    is_valid, reason, messages = validator.validate_conversation(record)
    assert is_valid is False
    assert "consecutive assistant" in reason.lower()


def test_validator_rejects_missing_assistant_response():
    validator = ConversationValidator()
    record = {
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }
    is_valid, reason, messages = validator.validate_conversation(record)
    assert is_valid is False
    assert "lacks required assistant response" in reason.lower()
