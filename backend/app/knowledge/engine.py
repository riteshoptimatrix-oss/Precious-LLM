"""
Precious Edu LLM — Project & Website Knowledge Engine Facade

Unified entry point coordinating KnowledgeRouter, ProjectKnowledgeSearch,
WebsiteKnowledgeService, and KnowledgeContextFormatter.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.knowledge.context import KnowledgeContextFormatter
from app.knowledge.models import KnowledgeHealthReport
from app.knowledge.repository import ProjectRepository
from app.knowledge.router import KnowledgeRouter, RoutingDecision
from app.knowledge.search import ProjectKnowledgeSearch
from app.knowledge.structured_search import StructuredQASearchEngine

logger = logging.getLogger(__name__)


class KnowledgeEngine:
    """
    Unified Knowledge Engine orchestrator coordinating:
    - Structured Q&A (PRIMARY knowledge source)
    - Website crawler chunks (SECONDARY supporting knowledge)
    - Project records (internal project metadata)
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        website_knowledge_service=None,
        structured_search=None,
    ):
        self.db = db
        self.repository = ProjectRepository(db)
        self.router = KnowledgeRouter()
        self.search_engine = ProjectKnowledgeSearch(db)
        self.formatter = KnowledgeContextFormatter()
        self.structured_search = structured_search or StructuredQASearchEngine(db)
        # Injected lazily to avoid circular imports at module load time
        self._website_service = website_knowledge_service

    def _get_website_service(self):
        """Returns the WebsiteKnowledgeService instance, constructing it lazily if needed."""
        if self._website_service is None:
            from app.website.services.website_knowledge_service import WebsiteKnowledgeService
            self._website_service = WebsiteKnowledgeService(self.db)
        return self._website_service

    async def retrieve_project_context(
        self,
        user_message: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[RoutingDecision, Optional[str]]:
        """
        Determines whether the message requires project knowledge.
        If yes, performs deterministic search and returns formatted context text.

        Returns:
            Tuple of (RoutingDecision, Optional[formatted_context_text])
        """
        decision, query = self.router.route(user_message, recent_messages)

        if decision == RoutingDecision.NO_KNOWLEDGE_REQUIRED or not query:
            return RoutingDecision.NO_KNOWLEDGE_REQUIRED, None

        if decision == RoutingDecision.WEBSITE_KNOWLEDGE_REQUIRED:
            return RoutingDecision.WEBSITE_KNOWLEDGE_REQUIRED, None

        # Execute MongoDB project search for project or combined decisions
        search_result = await self.search_engine.search(query)
        formatted_context = self.formatter.format_search_result(search_result)

        if decision == RoutingDecision.WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED:
            return RoutingDecision.WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED, formatted_context

        return RoutingDecision.PROJECT_KNOWLEDGE_REQUIRED, formatted_context

    async def retrieve_website_context(
        self,
        user_message: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[RoutingDecision, Optional[str]]:
        """
        Determines whether the message requires website knowledge.
        If yes, searches active website chunks and returns formatted context block.

        Returns:
            Tuple of (RoutingDecision, Optional[formatted_context_text])
        """
        decision, _ = self.router.route(user_message, recent_messages)

        if decision not in (
            RoutingDecision.WEBSITE_KNOWLEDGE_REQUIRED,
            RoutingDecision.WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED,
        ):
            return decision, None

        website_service = self._get_website_service()
        context_block = await website_service.get_context_block(user_message)
        return decision, context_block

    async def retrieve_combined_context(
        self,
        user_message: str,
        recent_messages: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[RoutingDecision, Optional[str], Optional[str], Optional[str], List[Any]]:
        """
        Unified multi-source knowledge retrieval:
        1. Classifies intent & resolves conversation follow-ups.
        2. Searches Structured Q&A as PRIMARY knowledge source.
        3. If structured match is strong, uses structured knowledge (and supporting website chunks if needed).
        4. If structured match is weak, falls back to website chunks as secondary.
        5. Searches project repository for explicit project queries (e.g. P001).

        Returns:
            Tuple of (RoutingDecision, project_context, website_context, structured_context, structured_candidates)
        """
        decision, query = self.router.route(user_message, recent_messages)
        intent_res = self.router.classify_intent(user_message, recent_messages)

        project_context: Optional[str] = None
        website_context: Optional[str] = None
        structured_context: Optional[str] = None
        structured_candidates: List[Any] = []

        if decision == RoutingDecision.NO_KNOWLEDGE_REQUIRED and intent_res.predicted_intent == "greeting":
            return decision, None, None, None, []

        # 1. PRIMARY: Search Structured Q&A
        try:
            structured_candidates = await self.structured_search.search(
                query_text=intent_res.resolved_query,
                intent=intent_res.predicted_intent
            )
            if structured_candidates:
                structured_context = self.structured_search.format_context_block(structured_candidates)
                decision = RoutingDecision.STRUCTURED_QA_REQUIRED
        except Exception as e:
            logger.warning(f"Structured Q&A search error (non-fatal): {e}")

        # 2. SECONDARY / SUPPORTING: Website Chunks
        # Fetch website context if structured is weak or if query domain warrants additional detail
        try:
            website_service = self._get_website_service()
            if website_service:
                should_fetch_web = (
                    not structured_candidates or
                    decision in (RoutingDecision.WEBSITE_KNOWLEDGE_REQUIRED, RoutingDecision.WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED) or
                    intent_res.predicted_intent in ["services", "privacy_policy", "terms_conditions"]
                )
                if should_fetch_web:
                    website_context = await website_service.get_context_block(
                        query_text=intent_res.resolved_query,
                        intent=intent_res.predicted_intent
                    )
        except Exception as e:
            logger.warning(f"Website knowledge search error (non-fatal): {e}")

        # 3. Project knowledge if requested
        if decision in (
            RoutingDecision.PROJECT_KNOWLEDGE_REQUIRED,
            RoutingDecision.WEBSITE_AND_PROJECT_KNOWLEDGE_REQUIRED,
        ) and query:
            try:
                search_result = await self.search_engine.search(query)
                project_context = self.formatter.format_search_result(search_result)
            except Exception as e:
                logger.warning(f"Project search error (non-fatal): {e}")

        if structured_context and website_context:
            decision = RoutingDecision.COMBINED_KNOWLEDGE_REQUIRED

        return decision, project_context, website_context, structured_context, structured_candidates

    async def get_health_status(self) -> KnowledgeHealthReport:
        """Fetch project knowledge system health status."""
        return await self.repository.get_health_status()
