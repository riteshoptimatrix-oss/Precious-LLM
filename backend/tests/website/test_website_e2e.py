import pytest
from unittest.mock import AsyncMock, MagicMock
from app.knowledge.router import KnowledgeRouter, RoutingDecision
from app.knowledge.engine import KnowledgeEngine
from app.conversation.context_builder import ContextBuilder, ConversationContext
from app.llm.service import LLMService

@pytest.mark.asyncio
async def test_e2e_website_knowledge_flow():
    # 1. Routing decision
    router = KnowledgeRouter()
    decision, _ = router.route("Tell me about student visa guidance for UK on Precious Education website.")
    assert decision == RoutingDecision.WEBSITE_KNOWLEDGE_REQUIRED

    # 2. Context retrieval simulation
    mock_website_service = MagicMock()
    mock_website_service.get_context_block = AsyncMock(return_value="[Source: UK Visa]\nCAS letter and financial proof are required for UK Tier 4 visa.")
    
    db = MagicMock()
    k_engine = KnowledgeEngine(db, website_knowledge_service=mock_website_service)
    decision, project_ctx, website_ctx, struct_ctx, candidates = await k_engine.retrieve_combined_context("student visa guidance for UK")
    
    assert website_ctx is not None
    assert "CAS letter and financial proof" in website_ctx

    # 3. Context builder assembly
    builder = ContextBuilder()
    context = builder.build_context(
        current_user_message="Tell me about student visa guidance for UK.",
        website_knowledge=website_ctx
    )
    
    assert context.website_knowledge == website_ctx

    # 4. LLM Service prompt serialization
    llm_service = LLMService()
    llm_service.tokenizer = MagicMock()
    llm_service.tokenizer.encode = lambda text: [1] * len(text)
    prompt = llm_service._serialize_context(context)
    
    assert "<website_knowledge>" in prompt
    assert "CAS letter and financial proof" in prompt
