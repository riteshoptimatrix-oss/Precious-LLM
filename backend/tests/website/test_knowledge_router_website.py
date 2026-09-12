import pytest
from app.knowledge.router import KnowledgeRouter, RoutingDecision

def test_router_website_query():
    router = KnowledgeRouter()
    decision, query = router.route("What services does Precious Education offer for studying in USA?")
    
    assert decision == RoutingDecision.WEBSITE_KNOWLEDGE_REQUIRED
    assert query is None

def test_router_combined_query():
    router = KnowledgeRouter()
    decision, query = router.route("What is the status of Project Alpha and what visa services do you offer for Canada?")
    
    assert decision == RoutingDecision.WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED
    assert query is not None

def test_router_chitchat():
    router = KnowledgeRouter()
    decision, query = router.route("Hello, how are you?")
    
    assert decision == RoutingDecision.NO_KNOWLEDGE_REQUIRED
    assert query is None

def test_router_project_only():
    router = KnowledgeRouter()
    decision, query = router.route("Who is the manager for P001?")
    
    assert decision == RoutingDecision.PROJECT_KNOWLEDGE_REQUIRED
    assert query.filters.get("project_id") == "P001"
