"""
Precious Edu LLM — Custom Tokenizer Exception Hierarchy

Defines custom exceptions for tokenizer training, serialization, encoding, and validation.
"""


class TokenizerError(Exception):
    """Base exception for all tokenizer errors."""
    pass


class TokenizerTrainingError(TokenizerError):
    """Raised when tokenizer training fails or data leakage rules are violated."""
    pass


class TokenizerSerializationError(TokenizerError):
    """Raised when saving or loading tokenizer artifacts fails."""
    pass


class InvalidVocabularyError(TokenizerError):
    """Raised when vocabulary integrity constraints (uniqueness, contiguity) are violated."""
    pass


class InvalidMergeRuleError(TokenizerError):
    """Raised when BPE merge rules are invalid or corrupted."""
    pass


class UnknownTokenError(TokenizerError):
    """Raised when an invalid token ID is referenced."""
    pass
