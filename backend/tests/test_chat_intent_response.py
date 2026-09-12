import pytest
from unittest.mock import AsyncMock, MagicMock
from app.llm.service import LLMService

def test_greeting_intent():
    service = LLMService()
    ctx = MagicMock()
    ctx.current_user_message = "Hello"
    ctx.website_knowledge = None
    ctx.project_knowledge = None

    response = service._clean_response("", context=ctx)
    assert "Hello! How can I assist you today?" in response

def test_services_question_intent():
    service = LLMService()
    ctx = MagicMock()
    ctx.current_user_message = "What services are you providing?"
    ctx.website_knowledge = "[Source: PEIC]\nPEIC is one of the reputed Overseas Education Consultants in India."
    ctx.project_knowledge = None

    response = service._clean_response("", context=ctx)
    assert "PEIC is one of the reputed Overseas Education Consultants" in response
    assert "Hello! How can I assist you today?" not in response

def test_ielts_question_intent():
    service = LLMService()
    ctx = MagicMock()
    ctx.current_user_message = "are provide the IELTS classes ?"
    ctx.website_knowledge = "[Source: PEIC]\nWe provide IELTS, TOEFL, GRE, and PTE coaching classes."
    ctx.project_knowledge = None

    response = service._clean_response("", context=ctx)
    assert "IELTS" in response
    assert "Hello! How can I assist you today?" not in response

def test_unknown_question_intent():
    service = LLMService()
    ctx = MagicMock()
    ctx.current_user_message = "What is the weather on Mars?"
    ctx.website_knowledge = None
    ctx.project_knowledge = None

    response = service._clean_response("", context=ctx)
    assert "knowledge base" in response.lower()
    assert "Hello! How can I assist you today?" not in response
