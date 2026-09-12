"""
Precious Edu LLM — Memory Unit Tests

Tests for memory extraction and management.
"""

import pytest

from app.services.memory_service import MemoryService


class TestMemoryExtraction:
    """Test rule-based memory extraction patterns."""

    def setup_method(self):
        """Set up a MemoryService for each test."""
        self.service = MemoryService()

    def test_extract_name_my_name_is(self):
        """Extract name from 'my name is X' pattern."""
        assert self.service.extract_user_name("My name is Ritesh") == "Ritesh"

    def test_extract_name_i_am(self):
        """Extract name from 'I am X' pattern."""
        assert self.service.extract_user_name("I am Ritesh") == "Ritesh"

    def test_extract_name_im(self):
        """Extract name from \"I'm X\" pattern."""
        assert self.service.extract_user_name("I'm Ritesh") == "Ritesh"

    def test_extract_name_call_me(self):
        """Extract name from 'call me X' pattern."""
        assert self.service.extract_user_name("Please call me Ritesh") == "Ritesh"

    def test_extract_name_case_insensitive(self):
        """Name extraction should be case insensitive."""
        assert self.service.extract_user_name("MY NAME IS ritesh") == "Ritesh"

    def test_no_name_in_message(self):
        """Return None when no name pattern is found."""
        assert self.service.extract_user_name("Hello, how are you?") is None

    def test_no_name_empty_message(self):
        """Return None for empty message."""
        assert self.service.extract_user_name("") is None

    def test_capitalizes_name(self):
        """Extracted name should be capitalized."""
        assert self.service.extract_user_name("my name is john") == "John"
