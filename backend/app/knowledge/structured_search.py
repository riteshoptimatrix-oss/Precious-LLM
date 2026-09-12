"""
Precious Edu LLM — Structured Q&A Search & Multi-Factor Scoring Engine

Implements deterministic multi-factor relevance ranking:
- Exact normalized query matching
- Substring phrase matching
- Weighted domain keyword overlap
- Intent alignment
- Legal / generic page de-ranking unless explicitly queried
- Generic company introduction de-ranking for specific topics
- Answerability scoring
"""

import logging
from typing import Any, Dict, List, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.core.config import get_settings
from app.knowledge.structured_qa_models import StructuredQACandidate
from app.knowledge.structured_qa_normalizer import StructuredQANormalizer
from app.knowledge.structured_repository import StructuredQARepository

logger = logging.getLogger(__name__)


class StructuredQASearchEngine:
    """
    Search and ranking engine for structured Q&A records.
    """

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        min_score: Optional[float] = None,
        top_k: Optional[int] = None
    ):
        settings = get_settings()
        self.db = db
        self.repo = StructuredQARepository(db)
        self.min_score = min_score if min_score is not None else settings.STRUCTURED_QA_MIN_SCORE
        self.top_k = top_k if top_k is not None else settings.STRUCTURED_QA_TOP_K

    def score_candidate(
        self,
        candidate: Dict[str, Any],
        raw_query: str,
        normalized_query: str,
        query_keywords: List[str],
        intent: Optional[str] = None
    ) -> StructuredQACandidate:
        """
        Computes multi-factor relevance score for a structured Q&A document.
        """
        c_input = candidate.get("user_input", "")
        c_norm_input = candidate.get("normalized_input", "")
        c_response = candidate.get("response", "")
        c_intent = candidate.get("intent", "")
        c_category = candidate.get("category", "")
        c_keywords = set(candidate.get("keywords", []))

        score_breakdown: Dict[str, float] = {}
        penalty_score = 0.0


        # 1. Exact match score
        exact_score = 0.0
        if normalized_query == c_norm_input:
            exact_score = 100.0
        elif raw_query.strip().lower() == c_input.strip().lower():
            exact_score = 100.0
        score_breakdown["exact_match"] = exact_score

        # 2. Phrase match score
        phrase_score = 0.0
        if normalized_query and len(normalized_query) >= 3:
            if normalized_query in c_norm_input:
                phrase_score += 45.0
            elif c_norm_input and c_norm_input in normalized_query:
                phrase_score += 35.0
            if normalized_query in c_response.lower():
                phrase_score += 15.0
        score_breakdown["phrase_match"] = phrase_score

        # 3. Intent alignment score
        intent_score = 0.0
        if intent and intent not in ("unknown", "greeting") and c_intent:
            if intent == c_intent:
                intent_score = 35.0
            elif intent in c_intent or c_intent in intent:
                intent_score = 20.0
        score_breakdown["intent_match"] = intent_score

        # 4. Domain keyword overlap score
        keyword_score = 0.0
        q_kw_set = set(query_keywords)
        common_kws = q_kw_set.intersection(c_keywords)

        for kw in common_kws:
            weight = StructuredQANormalizer.DOMAIN_WEIGHTS.get(kw, 1.5)
            keyword_score += (weight * 8.0)

        # Also check if query keywords appear in candidate response
        c_resp_lower = c_response.lower()
        for kw in q_kw_set:
            if kw in c_resp_lower and kw not in common_kws:
                weight = StructuredQANormalizer.DOMAIN_WEIGHTS.get(kw, 1.0)
                keyword_score += (weight * 4.0)

        score_breakdown["keyword_overlap"] = keyword_score

        # 4b. Country Entity Alignment
        q_country = StructuredQANormalizer.detect_country(raw_query) or StructuredQANormalizer.detect_country(normalized_query)
        c_full_text = f"{c_intent} {c_norm_input} {c_response} {c_category}".lower()
        c_country = StructuredQANormalizer.detect_country(c_full_text)

        country_score = 0.0
        if q_country:
            if c_country == q_country or (q_country.lower() in c_full_text):
                country_score += 45.0
            elif c_country and c_country != q_country:
                # Strong penalty for candidate matching wrong country
                penalty_score += 80.0
        score_breakdown["country_alignment"] = country_score

        # 4c. Visa Type Entity Alignment
        q_visa = StructuredQANormalizer.detect_visa_type(raw_query) or StructuredQANormalizer.detect_visa_type(normalized_query)
        c_visa = StructuredQANormalizer.detect_visa_type(c_full_text)

        visa_type_score = 0.0
        if q_visa:
            if c_visa == q_visa or (q_visa in c_intent):
                visa_type_score += 45.0
            elif c_visa and c_visa != q_visa:
                # If user asks for visitor visa, penalize student visa candidate
                if q_visa == "visitor" and c_visa == "student":
                    penalty_score += 90.0
                elif q_visa == "student" and c_visa == "visitor":
                    penalty_score += 90.0
                elif q_visa == "contact" and c_visa in ("student", "visitor", "work"):
                    penalty_score += 90.0
        score_breakdown["visa_type_alignment"] = visa_type_score

        # 4d. Specific Contact / Identity Intent Alignments
        if any(w in normalized_query for w in ["email", "mail id", "email address", "contact email"]):
            if c_intent == "contact_email":
                intent_score += 60.0
            elif c_intent.startswith("service_") or c_intent.startswith("countries_"):
                penalty_score += 80.0
        elif any(w in normalized_query for w in ["mobile", "phone", "mobile number", "phone number", "contact number"]):
            if c_intent == "contact_phone":
                intent_score += 60.0
            elif c_intent.startswith("service_") or c_intent.startswith("countries_"):
                penalty_score += 80.0
        elif any(w in normalized_query for w in ["who are you", "what are you"]):
            if c_intent in ("greetings", "about_us_general", "about_us_who_we_are"):
                intent_score += 60.0
            elif c_intent.startswith("service_"):
                penalty_score += 80.0
        elif any(w in normalized_query for w in ["spp", "ssp"]):
            if c_intent == "fullform_spp":
                intent_score += 60.0

        # 5. Penalties & Negative Signals


        # 5a. Legal page de-ranking (Privacy Policy & Terms and Conditions)
        is_legal_category = c_category in ["PrivacyPolicy", "TermsandConditions"] or c_intent.startswith("privacy_") or c_intent.startswith("terms_")
        is_legal_query = any(w in normalized_query for w in ["privacy", "policy", "terms", "condition", "conditions", "disclaimer", "cookie"])
        if is_legal_category and not is_legal_query:
            penalty_score += 500.0

        # 5b. Generic company intro de-ranking for specific queries
        is_specific_query = any(w in normalized_query for w in [
            "student visa", "study visa", "visitor visa", "tourist visa", "ielts", "coaching", "classes", "peic",
            "documents", "timing", "time", "duration", "canada", "australia", "uk", "usa", "email", "phone", "mobile"
        ])
        is_generic_intro = "established in 2005" in c_response.lower() or "management has total" in c_response.lower() or c_intent == "about_us_general"
        if is_specific_query and is_generic_intro and not any(w in normalized_query for w in ["who is the owner", "owner", "founder", "director", "history", "about"]):
            penalty_score += 45.0

        score_breakdown["penalties"] = -penalty_score

        # 6. Answerability score
        answerability = 1.0
        if len(c_response.strip()) < 15:
            answerability = 0.5
        score_breakdown["answerability"] = answerability

        total_score = (exact_score + phrase_score + intent_score + keyword_score + country_score + visa_type_score - penalty_score) * answerability

        return StructuredQACandidate(
            record_id=str(candidate.get("_id", "")),
            intent=c_intent,
            user_input=c_input,
            response=c_response,
            source_file=candidate.get("source_file", ""),
            category=c_category,
            score=max(0.0, total_score) if not (is_legal_category and not is_legal_query) else total_score,
            score_breakdown=score_breakdown,
            answerability=answerability,
        )

    async def search(
        self,
        query_text: str,
        intent: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[StructuredQACandidate]:
        """
        Executes end-to-end candidate retrieval and multi-factor ranking.
        """
        if not query_text or not query_text.strip():
            return []

        norm_query = StructuredQANormalizer.normalize_input(query_text)
        keywords = StructuredQANormalizer.extract_keywords(query_text)

        # Retrieve candidates from MongoDB (indexed)
        raw_candidates = await self.repo.get_candidates(
            normalized_query=norm_query,
            keywords=keywords,
            intent=intent,
            limit=200
        )

        if not raw_candidates:
            return []

        # Score candidates
        scored_candidates: List[StructuredQACandidate] = []
        for cand in raw_candidates:
            scored = self.score_candidate(
                candidate=cand,
                raw_query=query_text,
                normalized_query=norm_query,
                query_keywords=keywords,
                intent=intent
            )
            if scored.score >= self.min_score:
                scored_candidates.append(scored)

        # Sort descending by score
        scored_candidates.sort(key=lambda c: c.score, reverse=True)

        # Deduplicate candidates with identical response text to keep results diverse
        deduped: List[StructuredQACandidate] = []
        seen_responses = set()
        k_limit = top_k or self.top_k

        for c in scored_candidates:
            resp_snippet = c.response[:80].strip()
            if resp_snippet not in seen_responses:
                seen_responses.add(resp_snippet)
                deduped.append(c)
                if len(deduped) >= k_limit:
                    break

        return deduped

    def format_context_block(self, candidates: List[StructuredQACandidate]) -> Optional[str]:
        """
        Formats top candidates into a clean, structured XML-style context block for the LLM.
        """
        if not candidates:
            return None

        lines = ["The following verified Q&A knowledge is from Precious Education official records:\n"]
        for i, c in enumerate(candidates, start=1):
            lines.append(f"[Record {i}] (Intent: {c.intent}, Source: {c.source_file})")
            lines.append(f"Question: {c.user_input}")
            lines.append(f"Answer: {c.response}")
            lines.append("")

        return "\n".join(lines).strip()
