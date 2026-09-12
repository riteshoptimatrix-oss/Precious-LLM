"""
Precious Edu LLM — Unit Tests for Knowledge Intent Router

Tests chit-chat bypass, explicit project ID detection, keyword routing,
and multi-turn pronoun resolution.
"""

import pytest
from app.knowledge.router import KnowledgeRouter, RoutingDecision


def test_router_chit_chat_bypass():
    router = KnowledgeRouter()

    for msg in ["Hello", "hi", "Thanks", "bye", "What is my name?"]:
        decision, query = router.route(msg)
        assert decision == RoutingDecision.NO_KNOWLEDGE_REQUIRED
        assert query is None


def test_router_explicit_project_id():
    router = KnowledgeRouter()

    decision, query = router.route("Who manages P001?")

    assert decision == RoutingDecision.PROJECT_KNOWLEDGE_REQUIRED
    assert query is not None
    assert query.filters.get("project_id") == "P001"


def test_router_project_name_and_status():
    router = KnowledgeRouter()

    decision, query = router.route("What is Project Alpha's status?")

    assert decision == RoutingDecision.PROJECT_KNOWLEDGE_REQUIRED
    assert query is not None
    assert query.filters.get("project_name") == "Alpha"
    assert "status" in query.fields


def test_router_multi_turn_pronoun_resolution():
    router = KnowledgeRouter()

    recent_history = [
        {"role": "user", "content": "Tell me about Project Alpha."},
        {"role": "assistant", "content": "Project Alpha is currently in progress."}
    ]

    decision, query = router.route("Who manages it?", recent_messages=recent_history)

    assert decision == RoutingDecision.PROJECT_KNOWLEDGE_REQUIRED
    assert query is not None
    assert query.filters.get("project_name") == "Alpha"
    assert "manager" in query.fields
