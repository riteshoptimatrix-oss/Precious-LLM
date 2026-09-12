"""
Precious Edu LLM — Conversation System Policies & Instructions

Centralized definition of system instructions, assistant identity, and conversation policies.
"""

DEFAULT_SYSTEM_INSTRUCTIONS = (
    "You are Precious AI, a helpful, domain-specific conversational assistant. "
    "Maintain a respectful, conversational tone, leverage known user memory facts appropriately, "
    "and provide accurate responses without inventing false information."
)


class ConversationPolicy:
    """Policy rules governing conversation behavior, context limits, and memory rules."""

    SYSTEM_INSTRUCTIONS: str = DEFAULT_SYSTEM_INSTRUCTIONS
    ALLOW_MEMORY_EXTRACTION: bool = True
    STRICT_SESSION_ISOLATION: bool = True
