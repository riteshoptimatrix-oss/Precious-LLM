"""
Unit tests for LLMService in app.llm.service
"""

import pytest
import asyncio
from app.conversation.context_builder import ContextBuilder
from app.llm.config import LLMConfig
from app.llm.service import LLMService


@pytest.mark.asyncio
async def test_llm_service_generate():
    config = LLMConfig(deterministic=True, max_new_tokens=16)
    service = LLMService(config=config)
    service.initialize()

    assert service.is_ready() is True
    info = service.get_info()
    assert info["is_ready"] is True
    assert "parameters" in info

    builder = ContextBuilder()
    context = builder.build_context(
        current_user_message="Hello",
        memories={"user_name": "Ritesh"},
        recent_messages=[]
    )

    response = await service.generate(context)
    assert isinstance(response, str)
    assert len(response) > 0


def test_serialize_context_budget():
    config = LLMConfig()
    service = LLMService(config=config)
    service.initialize()

    builder = ContextBuilder()
    context = builder.build_context(
        current_user_message="What is a student visa?",
        memories={"user_name": "Ritesh"},
        recent_messages=[
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
        ]
    )

    prompt = service._serialize_context(context)
    assert "<system>" in prompt
    assert "<user>\nWhat is a student visa?" in prompt
    assert prompt.endswith("<assistant>\n")
