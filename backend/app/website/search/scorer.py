"""
Precious AI — Deterministic Website Relevance Scorer
"""

from typing import Dict, Any, List


class RelevanceScorer:
    """
    Computes deterministic relevance score for a website chunk against query terms and intent.
    """

    @classmethod
    def score_chunk(
        cls,
        chunk: Dict[str, Any],
        query_text: str,
        query_tokens: List[str],
        intent: str = None
    ) -> float:
        """
        Calculates relevance score based on title, heading, content term frequency, and intent alignment.
        """
        title = chunk.get("title", "").lower()
        section = chunk.get("section", "").lower()
        content = chunk.get("content", "").lower()
        url = chunk.get("canonical_url", "").lower()
        page_type = chunk.get("page_type", "").lower()

        is_legal_page = (
            "privacy-policy" in url or
            "terms-conditions" in url or
            "privacy policy" in title or
            "terms & conditions" in title or
            "terms and conditions" in title or
            page_type in ["legal", "privacy", "terms"]
        )

        # 1. Heavily penalize legal/policy pages if query/intent is NOT legal
        if is_legal_page:
            legal_intents = {"privacy_policy", "terms_conditions"}
            q_lower = query_text.lower()
            is_legal_query = any(w in q_lower for w in ["privacy", "policy", "terms", "condition", "disclaimer", "cookie"])
            if intent not in legal_intents and not is_legal_query:
                return -1000.0  # Excluded

        score = 0.0
        q_lower = query_text.lower().strip()

        # 2. Exact phrase match in title or section
        if q_lower in title:
            score += 50.0
        if q_lower in section:
            score += 30.0
        if q_lower in content:
            score += 15.0

        # 3. Intent specific boost
        if intent == "ielts_classes":
            if any(k in section for k in ["ielts", "coaching", "faculty", "faculties"]):
                score += 100.0
            if any(k in content for k in ["ielts", "coaching", "preparatory"]):
                score += 50.0
            if "index.html" in url or url.endswith("/"):
                score += 20.0

        elif intent == "services":
            if any(k in section for k in ["services", "counseling", "counselling", "consultant", "education"]):
                score += 80.0
            if "index.html" in url or url.endswith("/"):
                score += 40.0
            if "service" in content or "services" in content:
                score += 20.0

        elif intent == "visa_services":
            if any(k in section for k in ["visa", "student visa", "permit", "ratio"]):
                score += 150.0
            elif any(k in section for k in ["institutes", "universities", "colleges", "consultant", "education"]):
                score += 40.0
            if "index.html" in url or url.endswith("/"):
                score += 20.0
            if any(k in content for k in ["visa", "student visa", "study visa", "study abroad", "student permit"]):
                score += 50.0

        # 4. Token keyword occurrences
        for token in query_tokens:
            if not token or len(token) < 2:
                continue
            if token in title:
                score += 20.0
            if token in section:
                score += 10.0
            if token in content:
                count = content.count(token)
                score += min(count * 3.0, 15.0)

        return score
