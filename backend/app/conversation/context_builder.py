"""
Precious Edu LLM — Context Builder

Assembles a structured ConversationContext from multiple context sources
(system prompt, user memories, recent history window, current user message).
Decouples prompt structure from specific model tokenizers.
"""

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.conversation.policies import ConversationPolicy

logger = logging.getLogger(__name__)


class ConversationContext(BaseModel):
    """
    Structured context container sent to ResponseGenerator.
    """
    system_instructions: str = Field(..., description="System instructions and policy guidance")
    memories: Dict[str, str] = Field(default_factory=dict, description="Active user and session memories")
    recent_messages: List[Dict[str, Any]] = Field(default_factory=list, description="Recent conversation history turns")
    current_user_message: str = Field(..., description="The incoming user prompt for the current turn")
    project_knowledge: Optional[str] = Field(None, description="Retrieved project knowledge context block")
    website_knowledge: Optional[str] = Field(None, description="Retrieved website knowledge context block from preciousedu.in")
    structured_knowledge: Optional[str] = Field(None, description="Retrieved structured Q&A verified knowledge block")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Context metadata (budget limits, etc.)")


class ContextBuilder:
    """
    Assembles model context from system instructions, memories, history, structured Q&A, project knowledge, and current message.
    Enforces priority order:
    1. System instructions
    2. Relevant persistent memories
    3. Recent conversation
    4. Structured Q&A knowledge (primary)
    5. Website knowledge (supporting)
    6. Project knowledge (if retrieved)
    7. Current user message
    """

    def __init__(
        self,
        system_instructions: str = ConversationPolicy.SYSTEM_INSTRUCTIONS
    ):
        self.system_instructions = system_instructions

    def build_context(
        self,
        current_user_message: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None,
        memories: Optional[Dict[str, str]] = None,
        project_knowledge: Optional[str] = None,
        website_knowledge: Optional[str] = None,
        structured_knowledge: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationContext:
        """
        Build structured ConversationContext instance.

        Args:
            current_user_message: Incoming user text.
            recent_messages: Truncated list of recent message dicts ({role, content}).
            memories: Active session memories dict ({key: value}).
            project_knowledge: Formatted project facts context string.
            website_knowledge: Formatted website knowledge context string from preciousedu.in.
            structured_knowledge: Formatted structured Q&A verified knowledge string.
            metadata: Turn metadata.

        Returns:
            ConversationContext instance.
        """
        return ConversationContext(
            system_instructions=self.system_instructions,
            memories=memories or {},
            recent_messages=recent_messages or [],
            current_user_message=current_user_message,
            project_knowledge=project_knowledge,
            website_knowledge=website_knowledge,
            structured_knowledge=structured_knowledge,
            metadata=metadata or {}
        )

