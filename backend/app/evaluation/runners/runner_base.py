"""
Precious AI — Evaluation Base Runner

Provides core environment setup, deterministic LLM configuration, MongoDB test fixture connection,
inference timing, resource measurement, and structured failure classification.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.mongodb import connect_to_mongodb, close_mongodb_connection, get_client, get_database
from app.llm import get_llm_service, LLMService
from app.llm.config import LLMConfig
from app.services.session_service import SessionService
from app.conversation.engine import ConversationEngine
from app.evaluation.metrics.eval_metrics import EvaluationMetrics

logger = logging.getLogger(__name__)


class EvaluationRunnerBase:
    """
    Base class for evaluation layer runners.
    """

    def __init__(self, db_name: str = "precious_ai_eval"):
        self.db_name = db_name
        self.db: Optional[AsyncIOMotorDatabase] = None
        self.llm_service: Optional[LLMService] = None
        self.engine: Optional[ConversationEngine] = None
        self.session_service: Optional[SessionService] = None

    async def setup(self) -> None:
        """
        Initializes MongoDB test connection and configures LLMService in deterministic evaluation mode.
        """
        await connect_to_mongodb()
        client = get_client()
        self.db = client[self.db_name]

        # Configure deterministic LLM service
        llm_cfg = LLMConfig()
        llm_cfg.deterministic = True
        llm_cfg.temperature = 0.0
        llm_cfg.top_k = 1
        llm_cfg.top_p = 1.0

        self.llm_service = LLMService(config=llm_cfg)
        self.llm_service.initialize()

        self.engine = ConversationEngine(db=self.db, generator=self.llm_service)
        self.session_service = SessionService(self.db)

    async def teardown(self) -> None:
        """
        Cleans up test database and closes MongoDB connections.
        """
        if self.db is not None:
            client = get_client()
            if client is not None:
                await client.drop_database(self.db_name)
        await close_mongodb_connection()

    @staticmethod
    def classify_failure(error_msg: str, category_context: str = "") -> str:
        """
        Classifies evaluation failures into structured categories.
        Categories: MODEL, TOKENIZER, CONTEXT, MEMORY, DOMAIN, KNOWLEDGE, HALLUCINATION, GENERATION, API, DATABASE, SECURITY, PERFORMANCE, DATASET
        """
        err_lower = error_msg.lower()
        ctx_lower = category_context.lower()

        if "token" in err_lower or "vocab" in err_lower:
            return "TOKENIZER"
        elif "model" in err_lower or "weight" in err_lower or "checkpoint" in err_lower:
            return "MODEL"
        elif "memory" in err_lower or "session" in err_lower and "leak" in err_lower:
            return "MEMORY"
        elif "context" in err_lower or "history" in err_lower:
            return "CONTEXT"
        elif "project" in err_lower or "excel" in err_lower or "ground" in err_lower:
            return "KNOWLEDGE"
        elif "domain" in err_lower or "visa" in err_lower:
            return "DOMAIN"
        elif "hallucinat" in err_lower or "invent" in err_lower:
            return "HALLUCINATION"
        elif "mongo" in err_lower or "db" in err_lower or "query" in err_lower:
            return "DATABASE"
        elif "inject" in err_lower or "secur" in err_lower or "auth" in err_lower:
            return "SECURITY"
        elif "timeout" in err_lower or "slow" in err_lower or "latency" in err_lower:
            return "PERFORMANCE"
        elif "eos" in err_lower or "repeat" in err_lower or "clean" in err_lower:
            return "GENERATION"
        else:
            return "GENERAL"
