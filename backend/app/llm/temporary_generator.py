"""
Precious Edu LLM — Context-Aware Temporary Response Generator

Placeholder generator implementing ResponseGenerator.
Examines structured ConversationContext (memories, history, current message)
to produce contextually coherent responses for Phase 3 testing.
"""

from typing import Any
from app.llm.base import ResponseGenerator


class TemporaryResponseGenerator(ResponseGenerator):
    """
    Rule-based, context-aware generator used to validate the Phase 3 architecture
    prior to integrating the custom Transformer LLM in later phases.
    """

    async def generate(self, context: Any) -> str:
        msg = context.current_user_message.strip().lower() if context.current_user_message else ""
        memories = getattr(context, "memories", {}) or {}

        # 1. Name query matching
        if "what is my name" in msg or "who am i" in msg or "do you know my name" in msg or "my name?" in msg:
            user_name = memories.get("user_name")
            if user_name:
                return f"Your name is {user_name}."
            return "I don't know your name yet. What should I call you?"

        # 2. Name statement matching (for response phrasing)
        if any(msg.startswith(p) for p in ["my name is", "i am ", "i'm ", "call me "]):
            user_name = memories.get("user_name")
            if user_name:
                return f"Nice to meet you, {user_name}!"

        # 3. Standard conversation triggers
        if "hello" in msg or "hi" in msg or "hey" in msg:
            user_name = memories.get("user_name")
            if user_name:
                return f"Hello {user_name}! Welcome to Precious AI. How can I assist you today?"
            return "Hello! Welcome to Precious AI. How can I assist you today?"
        elif "thanks" in msg or "thank you" in msg:
            return "You're very welcome!"
        elif "bye" in msg or "goodbye" in msg:
            return "Goodbye! Have a great day!"

        # 4. Fallback response with context awareness
        user_name = memories.get("user_name")
        if user_name:
            return f"Thank you for your message, {user_name}. [Phase 3 Engine Verified]"
        return f"Thank you for your message. [Phase 3 Engine Verified]"
