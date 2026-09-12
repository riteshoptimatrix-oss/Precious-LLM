"""
Precious Edu LLM — Prompt Templates

Defines the prompt format templates used to assemble model inputs.

These templates define how different context blocks (system prompt,
memory, knowledge, history, user message) are formatted and
combined into the final prompt string.
"""


class PromptTemplates:
    """
    Collection of prompt format templates.

    Using special tokens:
    - <BOS>  : Beginning of sequence
    - <EOS>  : End of sequence
    - <SEP>  : Separator between blocks
    - <SYS>  : System prompt marker
    - <USER> : User turn marker
    - <ASST> : Assistant turn marker
    """

    # Default system prompt for the chatbot
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful and friendly assistant for Precious Edu. "
        "Answer questions clearly and concisely. "
        "If you don't know something, say so honestly."
    )

    # Template for the full prompt with all blocks
    FULL_PROMPT = (
        "<BOS>"
        "<SYS>{system_prompt}<SEP>"
        "{memory_block}"
        "{knowledge_block}"
        "{history_block}"
        "<USER>{user_message}<SEP>"
        "<ASST>"
    )

    # Template for memory block (included only when memory exists)
    MEMORY_BLOCK = "{memory_text}<SEP>"

    # Template for knowledge block (included only when knowledge is retrieved)
    KNOWLEDGE_BLOCK = "Relevant information: {knowledge_text}<SEP>"

    # Template for a single conversation turn
    TURN_USER = "<USER>{content}<SEP>"
    TURN_ASSISTANT = "<ASST>{content}<SEP>"

    @classmethod
    def format_prompt(
        cls,
        user_message: str,
        system_prompt: str | None = None,
        memory_text: str = "",
        knowledge_text: str = "",
        history_text: str = "",
    ) -> str:
        """
        Format a complete prompt from all blocks.

        Args:
            user_message: Current user message.
            system_prompt: System instruction. Uses default if None.
            memory_text: Formatted memory block text.
            knowledge_text: Formatted knowledge block text.
            history_text: Formatted conversation history text.

        Returns:
            Complete formatted prompt string.
        """
        sys_prompt = system_prompt or cls.DEFAULT_SYSTEM_PROMPT

        # Only include memory/knowledge blocks if they have content
        memory_block = cls.MEMORY_BLOCK.format(memory_text=memory_text) if memory_text else ""
        knowledge_block = cls.KNOWLEDGE_BLOCK.format(knowledge_text=knowledge_text) if knowledge_text else ""

        return cls.FULL_PROMPT.format(
            system_prompt=sys_prompt,
            memory_block=memory_block,
            knowledge_block=knowledge_block,
            history_block=history_text,
            user_message=user_message,
        )

    @classmethod
    def format_history(cls, messages: list) -> str:
        """
        Format a list of messages into history text.

        Args:
            messages: List of {role, content} dicts.

        Returns:
            Formatted history string.
        """
        parts = []
        for msg in messages:
            if msg["role"] == "user":
                parts.append(cls.TURN_USER.format(content=msg["content"]))
            elif msg["role"] == "assistant":
                parts.append(cls.TURN_ASSISTANT.format(content=msg["content"]))
        return "".join(parts)
