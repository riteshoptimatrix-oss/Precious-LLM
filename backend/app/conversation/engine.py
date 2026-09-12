"""
Precious Edu LLM — Conversation Engine

Central orchestrator for chat turns.
Coordinates session validation, user message persistence, context budgeting,
memory extraction & retrieval, model generation, assistant message persistence,
and response formatting.

Phase 13: Extended to retrieve website knowledge from preciousedu.in and
inject it into ConversationContext alongside project knowledge.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.exceptions import SessionNotFoundError
from app.models.message import MessageModel, MessageRole
from app.repositories.session_repository import SessionRepository
from app.repositories.message_repository import MessageRepository
from app.repositories.memory_repository import MemoryRepository
from app.conversation.context_manager import ContextManager
from app.conversation.context_builder import ContextBuilder
from app.conversation.memory_manager import MemoryManager
from app.knowledge.engine import KnowledgeEngine
from app.knowledge.router import RoutingDecision
from app.llm.base import ResponseGenerator
from app.llm.temporary_generator import TemporaryResponseGenerator

logger = logging.getLogger(__name__)


class ConversationEngine:
    """
    Central orchestration engine for conversation processing.
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        generator: Optional[ResponseGenerator] = None,
        context_manager: Optional[ContextManager] = None,
        context_builder: Optional[ContextBuilder] = None,
        memory_manager: Optional[MemoryManager] = None,
        knowledge_engine: Optional[KnowledgeEngine] = None,
        website_knowledge_service=None,
    ):
        self.db = db
        self.session_repo = SessionRepository(db)
        self.message_repo = MessageRepository(db)
        self.memory_repo = MemoryRepository(db)
        if generator is not None:
            self.generator = generator
        else:
            try:
                from app.llm import get_llm_service
                self.generator = get_llm_service()
            except Exception as e:
                logger.warning(f"Could not initialize LLMService: {e}")
                self.generator = TemporaryResponseGenerator()
        self.context_manager = context_manager or ContextManager(self.message_repo)

        self.context_builder = context_builder or ContextBuilder()
        self.memory_manager = memory_manager or MemoryManager(self.memory_repo)

        # Inject website_knowledge_service into KnowledgeEngine for combined routing
        _website_svc = website_knowledge_service or self._build_website_service()
        self.knowledge_engine = knowledge_engine or KnowledgeEngine(
            db=db,
            website_knowledge_service=_website_svc,
        )
        self._website_service = _website_svc

    def _build_website_service(self):
        """Lazily constructs WebsiteKnowledgeService to avoid startup failures
        when website index hasn't been populated yet."""
        try:
            from app.website.services.website_knowledge_service import WebsiteKnowledgeService
            return WebsiteKnowledgeService(self.db)
        except Exception as e:
            logger.warning(f"Could not initialize WebsiteKnowledgeService: {e}")
            return None

    async def handle_message(
        self,
        session_id: str,
        user_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate a complete conversation turn:
        1.  Validate session existence.
        2.  Persist incoming user message in MongoDB.
        3.  Extract memory facts from user message & update MemoryRepository (fault-isolated).
        4.  Load recent context history (budgeted via ContextManager).
        5.  Fetch active session memories (via MemoryManager).
        6.  Evaluate intent & retrieve project knowledge if required (via KnowledgeEngine).
        6b. Retrieve website knowledge if required (via KnowledgeEngine / WebsiteKnowledgeService).
        7.  Construct structured ConversationContext (via ContextBuilder).
        8.  Invoke ResponseGenerator interface.
        9.  Persist assistant response in MongoDB.
        10. Update session timestamp.
        11. Return response payload including source references.
        """
        # 1. Validate session
        session = await self.session_repo.get_by_session_id(session_id)
        if not session:
            raise SessionNotFoundError(session_id)

        # 2. Persist user message
        user_msg_model = MessageModel(
            session_id=session_id,
            role=MessageRole.USER,
            content=user_message,
            metadata=metadata or {}
        )
        saved_user_msg = await self.message_repo.create(user_msg_model)
        user_msg_id = str(saved_user_msg.get("_id", "")) if isinstance(saved_user_msg, dict) else None

        # 3. Extract & persist memories (fault-isolated)
        await self.memory_manager.process_user_message(
            session_id=session_id,
            user_message=user_message,
            source_message_id=user_msg_id
        )

        # 4. Load recent history
        recent_history = await self.context_manager.get_recent_messages(session_id)

        # 5. Fetch active session memories
        active_memories = await self.memory_manager.get_memories_dict(session_id)

        # 6. Intent Classification & Multi-Source Knowledge Retrieval
        intent_res = self.knowledge_engine.router.classify_intent(user_message, recent_history)
        proj_context_text: Optional[str] = None
        web_context_text: Optional[str] = None
        struct_context_text: Optional[str] = None
        structured_candidates: List[Any] = []
        website_candidates: List[Dict[str, Any]] = []
        website_sources: List[Dict[str, str]] = []
        routing_decision = RoutingDecision.NO_KNOWLEDGE_REQUIRED

        try:
            (
                routing_decision,
                proj_context_text,
                web_context_text,
                struct_context_text,
                structured_candidates,
            ) = await self.knowledge_engine.retrieve_combined_context(
                user_message=user_message,
                recent_messages=recent_history,
            )
        except Exception as e:
            logger.warning(f"Knowledge retrieval error (non-fatal): {e}")

        # Fetch website candidate details for logging & secondary sources
        if self._website_service and intent_res:
            try:
                website_candidates = await self._website_service.search(
                    query_text=intent_res.resolved_query,
                    intent=intent_res.predicted_intent
                )
                website_sources = self._website_service.get_sources(website_candidates)
            except Exception as e:
                logger.warning(f"Website candidate retrieval error (non-fatal): {e}")

        # 7. Knowledge Ranking, Selection & Grounding
        from app.core.config import get_settings
        settings = get_settings()

        predicted_intent = intent_res.predicted_intent if intent_res else "unknown"
        confidence = intent_res.confidence if intent_res else 0.20
        resolved_query = intent_res.resolved_query if intent_res else user_message
        previous_context = intent_res.previous_context if intent_res else None

        direct_threshold = getattr(settings, "STRUCTURED_QA_DIRECT_ANSWER_THRESHOLD", 45.0)
        min_struct_score = getattr(settings, "STRUCTURED_QA_MIN_SCORE", 20.0)

        response_source = "fallback"
        knowledge_sources = []
        selected_id = "None"
        selected_intent = predicted_intent
        selected_source = "None"
        selected_title = "None"
        selected_score = "None"
        response_text = ""

        # Check Greeting First
        if predicted_intent == "greeting":
            response_source = "greeting"
            selected_id = "greeting-0"
            selected_intent = "greeting"
            selected_source = "system"
            selected_title = "Greeting"
            selected_score = "1.0"
            context = self.context_builder.build_context(
                current_user_message=resolved_query,
                recent_messages=recent_history,
                memories=active_memories,
                project_knowledge=proj_context_text,
                website_knowledge=web_context_text,
                structured_knowledge=None,
                metadata=metadata or {}
            )
            response_text = await self.generator.generate(context)
            if not response_text:
                response_text = "Hello! How can I assist you today?"

        # PRIMARY: Strong Structured Q&A Match
        elif structured_candidates and structured_candidates[0].score >= min_struct_score:
            top_sq = structured_candidates[0]

            # Avoid verbatim repetition on follow-ups (e.g. "tell me how")
            recent_assistant_responses = [
                m.get("content", "").strip()
                for m in (recent_history or [])
                if isinstance(m, dict) and m.get("role") == "assistant"
            ]
            if len(structured_candidates) > 1 and any(
                top_sq.response.strip() in r or r in top_sq.response.strip()
                for r in recent_assistant_responses[-2:]
                if r
            ):
                for next_cand in structured_candidates[1:]:
                    if next_cand.score >= min_struct_score and not any(
                        next_cand.response.strip() in r or r in next_cand.response.strip()
                        for r in recent_assistant_responses[-2:]
                        if r
                    ):
                        top_sq = next_cand
                        break

            selected_id = top_sq.record_id
            selected_intent = top_sq.intent
            selected_source = top_sq.source_file
            selected_title = f"Structured QA — {top_sq.category}"
            selected_score = f"{top_sq.score:.1f}"
            response_source = "structured_qa"
            knowledge_sources = [{
                "type": "structured_qa",
                "intent": top_sq.intent,
                "source_file": top_sq.source_file,
                "category": top_sq.category,
                "score": f"{top_sq.score:.1f}",
                "user_input": top_sq.user_input,
            }]

            # Generate grounded, diverse response via the Custom LLM Service
            candidate_struct_text = struct_context_text or (
                f"[Record: {top_sq.record_id}]\n"
                f"Intent: {top_sq.intent}\n"
                f"Category: {top_sq.category}\n"
                f"Input: {top_sq.user_input}\n"
                f"Answer: {top_sq.response}"
            )
            context = self.context_builder.build_context(
                current_user_message=resolved_query,
                recent_messages=recent_history,
                memories=active_memories,
                project_knowledge=proj_context_text,
                website_knowledge=web_context_text,
                structured_knowledge=candidate_struct_text,
                metadata=metadata or {}
            )
            response_text = await self.generator.generate(context)
            if not response_text:
                response_text = top_sq.response

        # SECONDARY: Website Chunks Match
        elif (web_context_text or (website_candidates and website_candidates[0].get("score", 0.0) >= getattr(settings, "WEBSITE_MIN_SCORE", 5.0))):
            top_wc = website_candidates[0] if website_candidates else {}
            selected_id = str(top_wc.get("_id", "chunk-0"))
            selected_intent = top_wc.get("intent") or predicted_intent
            selected_source = top_wc.get("canonical_url", "https://www.preciousedu.in/")
            selected_title = top_wc.get("title", "Precious Education")
            selected_score = f"{top_wc.get('score', 0.0):.1f}"
            response_source = "website_chunks"
            knowledge_sources = website_sources

            context = self.context_builder.build_context(
                current_user_message=resolved_query,
                recent_messages=recent_history,
                memories=active_memories,
                project_knowledge=proj_context_text,
                website_knowledge=web_context_text,
                structured_knowledge=None,
                metadata=metadata or {}
            )
            response_text = await self.generator.generate(context)

        # FALLBACK: No reliable candidate in structured_qa or website_chunks
        else:
            response_source = "fallback"
            context = self.context_builder.build_context(
                current_user_message=resolved_query,
                recent_messages=recent_history,
                memories=active_memories,
                project_knowledge=proj_context_text,
                website_knowledge=web_context_text,
                structured_knowledge=None,
                metadata=metadata or {}
            )
            generated_text = await self.generator.generate(context)
            if generated_text and "[Phase 3 Engine Verified]" not in generated_text:
                response_text = generated_text
            else:
                response_text = (
                    "I'm sorry, I don't have information about that in my knowledge base. "
                    "How else can I help you with Precious Education services or projects?"
                )

        # Fallback safeguard if generator produced blank or placeholder text
        if not response_text or "[Phase 3 Engine Verified]" in response_text:
            if response_source == "greeting":
                response_text = "Hello! How can I assist you today?"
            elif response_source == "structured_qa" and structured_candidates:
                response_text = top_sq.response
            else:
                response_text = (
                    "I'm sorry, I don't have information about that in my knowledge base. "
                    "How else can I help you with Precious Education services or projects?"
                )

        # Clean repetitive template boilerplate & promotional URLs
        from app.knowledge.structured_qa_normalizer import StructuredQANormalizer
        if response_text:
            response_text = StructuredQANormalizer.clean_boilerplate(response_text)


        # Section 24: Critical Development Debug Output
        gen_cfg = getattr(self.generator, "config", None)
        ckpt_name = getattr(self.generator, "checkpoint_path", None)
        logger.info("\n" + "=" * 40)
        logger.info("PRECIOUS AI DEBUG")
        logger.info("=================")
        logger.info(f"QUERY: {user_message}")
        logger.info(f"INTENT: {predicted_intent}")
        logger.info(f"COUNTRY: {intent_res.country_entity if intent_res else 'None'}")
        logger.info(f"SERVICE: {intent_res.visa_type if intent_res else 'None'}")
        logger.info(f"WEBSITE CHUNKS: {len(website_candidates)} chunks retrieved")
        logger.info(f"PROJECT CONTEXT: {'Yes' if proj_context_text else 'None'}")
        logger.info(f"CONVERSATION CONTEXT: {len(recent_history)} prior turns")
        logger.info(f"MODEL CHECKPOINT: {ckpt_name or 'backend/artifacts/fine_tuning/latest'}")
        logger.info(f"TEMPERATURE: {getattr(gen_cfg, 'temperature', 0.65)}")
        logger.info(f"TOP_K: {getattr(gen_cfg, 'top_k', 40)}")
        logger.info(f"TOP_P: {getattr(gen_cfg, 'top_p', 0.90)}")
        logger.info(f"REPETITION_PENALTY: {getattr(gen_cfg, 'repetition_penalty', 1.10)}")
        logger.info(f"GENERATED TOKENS: {len(response_text.split())} words")
        logger.info(f"FINAL RESPONSE: {response_text}")
        logger.info("=" * 40 + "\n")


        # 8. Persist assistant response
        assistant_msg_model = MessageModel(
            session_id=session_id,
            role=MessageRole.ASSISTANT,
            content=response_text,
            metadata={
                "generator": self.generator.__class__.__name__,
                "memories_used": list(active_memories.keys()),
                "routing_decision": routing_decision.value,
                "response_source": response_source,
                "source_type": response_source,
                "selected_intent": selected_intent,
                "selected_record_id": selected_id,
                "selected_source_file": selected_source,
                "relevance_score": selected_score,
                "predicted_intent": predicted_intent,
                "confidence": confidence,
                "sources": knowledge_sources or website_sources,
            }
        )
        saved_assistant_msg = await self.message_repo.create(assistant_msg_model)

        # 9. Update session timestamp
        await self.session_repo.update_timestamp(session_id)

        # 10. Format and return response
        created_at = saved_assistant_msg.get("created_at")
        if isinstance(created_at, datetime):
            created_at_str = created_at.isoformat()
        else:
            created_at_str = str(created_at)

        return {
            "session_id": session_id,
            "response": response_text,
            "role": MessageRole.ASSISTANT.value,
            "created_at": created_at_str,
            "sources": knowledge_sources or website_sources,
            "metadata": saved_assistant_msg.get("metadata", {})
        }
