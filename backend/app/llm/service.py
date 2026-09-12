"""
Precious Edu LLM — LLM Service Implementation

Implements ResponseGenerator ABC interfacing the ConversationEngine with the custom Phase 8 Transformer model.
Loads the model once on application startup, performs context budget prioritization,
tokenization, autoregressive text generation, and response post-processing.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from app.llm.base import ResponseGenerator
from app.llm.config import LLMConfig
from app.llm.loader import CustomLLMModelLoader
from app.ml.inference.generator import TextGenerator
from app.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class LLMService(ResponseGenerator):
    """
    Service facade encapsulating Phase 8 Custom LLM inference.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.loader = CustomLLMModelLoader(self.config)

        self.model = None
        self.tokenizer = None
        self.model_config = None
        self.generator = None
        self.checkpoint_path = None
        self.is_loaded = False

    def initialize(self) -> None:
        """Loads tokenizer and model weights into memory ONCE during application startup."""
        if self.is_loaded:
            logger.info("LLMService is already initialized.")
            return

        logger.info("Initializing LLMService...")
        self.model, self.tokenizer, self.model_config, self.checkpoint_path = self.loader.load()

        st_map = self.tokenizer.config.special_tokens_map
        self.generator = TextGenerator(
            model=self.model,
            eos_token_id=st_map[self.tokenizer.config.EOS_TOKEN],
            pad_token_id=st_map[self.tokenizer.config.PAD_TOKEN],
        )

        # Optional Startup Warmup Pass
        if self.config.warmup_on_startup:
            self._warmup()

        self.is_loaded = True
        logger.info("LLMService initialization complete.")

    def _warmup(self) -> None:
        """Executes a single dummy forward pass to warm up PyTorch CUDA/CPU kernels."""
        try:
            bos_id = self.tokenizer.config.special_tokens_map[self.tokenizer.config.BOS_TOKEN]
            dummy_input = torch.tensor([[bos_id]], dtype=torch.long, device=next(self.model.parameters()).device)
            _ = self.generator.generate(dummy_input, max_new_tokens=2, temperature=0.0)
            logger.info("LLM kernel warmup completed.")
        except Exception as e:
            logger.warning(f"LLM kernel warmup warning: {e}")

    def is_ready(self) -> bool:
        """Returns True if the LLM model and tokenizer are loaded in memory."""
        return self.is_loaded and self.model is not None and self.tokenizer is not None

    def get_info(self) -> Dict[str, Any]:
        """Returns diagnostic metadata about the loaded model."""
        if not self.is_ready():
            return {
                "is_ready": False,
                "status": "unloaded",
            }

        device = str(next(self.model.parameters()).device)
        num_params = self.model.num_parameters

        return {
            "is_ready": True,
            "status": "ready",
            "device": device,
            "model_version": getattr(self.model_config, "model_version", "1.0.0"),
            "tokenizer_version": self.tokenizer.config.TOKENIZER_VERSION,
            "vocab_size": self.tokenizer.vocab_size,
            "max_seq_length": getattr(self.model_config, "max_seq_length", 64),
            "parameters": num_params,
            "checkpoint": str(self.checkpoint_path.name) if self.checkpoint_path else "unknown",
        }

    def _serialize_context(self, context: Any) -> str:
        """
        Serializes ConversationContext into a tokenizable prompt with priority bounds.
        Priority order:
        1. System instructions
        2. Relevant memories
        3. Current user message
        4. Recent conversation history (truncated from oldest turns to fit max_context_length)
        """
        system_instructions = getattr(context, "system_instructions", "") or "You are Precious AI."
        memories = getattr(context, "memories", {}) or {}
        recent_messages = getattr(context, "recent_messages", []) or []
        current_user_message = getattr(context, "current_user_message", "") or ""
        project_knowledge = getattr(context, "project_knowledge", None)
        website_knowledge = getattr(context, "website_knowledge", None)
        structured_knowledge = getattr(context, "structured_knowledge", None)

        # Build System Header
        sys_parts = [system_instructions]
        if memories:
            mem_str = ", ".join(f"{k}: {v}" for k, v in memories.items())
            sys_parts.append(f"[Memories: {mem_str}]")

        system_text = f"<system>\n{' '.join(sys_parts)}"
        user_text = f"<user>\n{current_user_message.strip()}"
        asst_header = "<assistant>\n"

        proj_text = f"<project_knowledge>\n{project_knowledge.strip()}" if isinstance(project_knowledge, str) and project_knowledge.strip() else None
        web_text = f"<website_knowledge>\n{website_knowledge.strip()}" if isinstance(website_knowledge, str) and website_knowledge.strip() else None
        struct_text = f"<structured_knowledge>\n{structured_knowledge.strip()}" if isinstance(structured_knowledge, str) and structured_knowledge.strip() else None

        max_ctx = getattr(self.model_config, "max_seq_length", self.config.max_context_length)

        # Budget calculation
        # Reserve room for system_text, knowledge texts (if present), user_text, asst_header
        history_turns = []
        for msg in reversed(recent_messages):
            role = msg.get("role", "").lower().strip()
            content = msg.get("content", "").strip()
            if not content:
                continue

            if role == "user":
                turn_str = f"<user>\n{content}"
            elif role == "assistant":
                turn_str = f"<assistant>\n{content}"
            else:
                continue

            # Candidate prompt check
            base_parts = [system_text]
            extra_parts = [p for p in [struct_text, proj_text, web_text] if p]
            if extra_parts:
                candidate_prompt = "\n".join(base_parts + [turn_str] + history_turns + extra_parts + [user_text, asst_header])
            else:
                candidate_prompt = "\n".join(base_parts + [turn_str] + history_turns + [user_text, asst_header])

            encoded_len = len(self.tokenizer.encode(candidate_prompt))

            if encoded_len <= (max_ctx - 10):
                history_turns.insert(0, turn_str)
            else:
                break

        full_prompt_parts = [system_text] + history_turns
        for extra in [struct_text, proj_text, web_text]:
            if extra:
                full_prompt_parts.append(extra)
        full_prompt_parts.extend([user_text, asst_header])

        return "\n".join(full_prompt_parts)


    def _sync_generate(self, context: Any) -> str:
        """
        Synchronous model generation method run inside a worker thread.
        """
        if not self.is_ready():
            self.initialize()

        prompt_str = self._serialize_context(context)
        prompt_ids = self.tokenizer.encode(prompt_str)

        device = next(self.model.parameters()).device
        input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)

        temp = 0.0 if self.config.deterministic else self.config.temperature
        top_k = 1 if self.config.deterministic else self.config.top_k

        # Run Autoregressive Generation
        output_tensor = self.generator.generate(
            input_ids=input_tensor,
            max_new_tokens=self.config.max_new_tokens,
            temperature=temp,
            top_k=top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
        )

        gen_ids = output_tensor[0, len(prompt_ids):].tolist()
        raw_decoded = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Clean generated text passing context for intent-aware fallback
        cleaned = self._clean_response(raw_decoded, context=context)
        return cleaned

    def _is_greeting(self, message: str) -> bool:
        msg = message.strip().lower()
        exact_greetings = {"hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening", "good day"}
        if msg in exact_greetings or any(msg.startswith(g) for g in ["hello ", "hi ", "hey ", "good morning ", "good evening "]):
            return True
        return False

    def _extract_knowledge_answer(self, context: Any) -> Optional[str]:
        """Extracts clean answer text from retrieved structured Q&A, website, or project knowledge blocks."""
        structured_knowledge = getattr(context, "structured_knowledge", None)
        web_knowledge = getattr(context, "website_knowledge", None)
        proj_knowledge = getattr(context, "project_knowledge", None)

        if isinstance(structured_knowledge, str) and structured_knowledge.strip():
            clean_text = structured_knowledge.replace("<structured_knowledge>", "").strip()
            # Look for Answer: section
            answer_parts = []
            for block in clean_text.split("[Record"):
                if not block.strip():
                    continue
                if "Answer:" in block:
                    ans = block.split("Answer:", 1)[1].strip()
                    # Take up to the next section or end
                    if ans:
                        answer_parts.append(ans)
            if answer_parts:
                return answer_parts[0]
            return clean_text

        if isinstance(web_knowledge, str) and web_knowledge.strip():
            clean_text = web_knowledge.replace("<website_knowledge>", "").strip()
            lines = [l.strip() for l in clean_text.splitlines() if l.strip()]
            content_lines = [
                l for l in lines
                if not l.startswith("[Source") and
                not l.startswith("===") and
                not l.startswith("The following information is from")
            ]
            if content_lines:
                # Exclude general intro lines if specific topic/service content lines are available
                specific_lines = [
                    l for l in content_lines
                    if not l.startswith("Precious Education and Immigration Consultant was established in 2005") and
                    not l.startswith("The Precious Education and Immigration Consultant Management has total")
                ]
                target_lines = specific_lines if specific_lines else content_lines
                joined = " ".join(target_lines[:2])
                if len(joined) > 400:
                    joined = joined[:400].rsplit(".", 1)[0] + "."
                return joined
            return clean_text

        if proj_knowledge and proj_knowledge.strip():
            clean_text = proj_knowledge.replace("<project_knowledge>", "").strip()
            return clean_text

        return None

    def _clean_response(self, text: str, context: Optional[Any] = None) -> str:
        """Cleans control tokens, leading role markers, and formatting artifacts."""
        user_msg = getattr(context, "current_user_message", "") if context else ""
        
        clean_lines = []
        if text:
            for line in text.splitlines():
                l = line.strip()
                if l.startswith("<user>") or l.startswith("<system>") or l.startswith("<assistant>"):
                    continue
                if l in ("<eos>", "<pad>", "<unk>"):
                    continue
                clean_lines.append(line)

        result = "\n".join(clean_lines).strip()

        # If model generated valid non-empty response, return it
        if result and len(result) > 3 and result not in ("<eos>", "<pad>", "<unk>"):
            return result

        # Fallback handling based on context and message intent
        if self._is_greeting(user_msg):
            return "Hello! How can I assist you today?"

        # Extract answer from knowledge context if available
        if context:
            kn_answer = self._extract_knowledge_answer(context)
            if kn_answer:
                return kn_answer

        # Informative fallback for questions not in knowledge base
        return "I'm sorry, I don't have information about that in my knowledge base. How else can I help you with Precious Education services or projects?"

    async def generate(self, context: Any) -> str:
        """
        Asynchronously generates a response for a ConversationContext.
        Executes model matrix multiplication inside asyncio.to_thread.
        """
        if not self.is_ready():
            self.initialize()

        return await asyncio.to_thread(self._sync_generate, context)


# Singleton LLM Service Instance
_llm_service_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get global LLMService singleton instance."""
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance
