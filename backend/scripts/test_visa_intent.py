import asyncio
import os
import sys
import re
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongodb, get_database
from app.knowledge.router import KnowledgeRouter
from app.website.search.query import WebsiteQueryNormalizer

class RelevanceScorerTest:
    @classmethod
    def score_chunk(
        cls,
        chunk: Dict[str, Any],
        query_text: str,
        query_tokens: List[str],
        intent: Optional[str] = None
    ) -> float:
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

        if is_legal_page:
            legal_intents = {"privacy_policy", "terms_conditions"}
            q_lower = query_text.lower()
            is_legal_query = any(w in q_lower for w in ["privacy", "policy", "terms", "condition", "disclaimer", "cookie"])
            if intent not in legal_intents and not is_legal_query:
                return -1000.0

        score = 0.0
        q_lower = query_text.lower().strip()

        if q_lower in title:
            score += 50.0
        if q_lower in section:
            score += 30.0
        if q_lower in content:
            score += 15.0

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
            if any(k in section for k in ["visa", "ratio", "institutes", "education", "consultant"]):
                score += 80.0
            if "index.html" in url or url.endswith("/"):
                score += 40.0
            if any(k in content for k in ["visa", "student visa", "study abroad", "universities", "colleges"]):
                score += 30.0

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

class ValidatorTest:
    @classmethod
    def validate_chunk(cls, chunk: Dict[str, Any], intent: Optional[str]) -> bool:
        url = chunk.get("canonical_url", "").lower()
        title = chunk.get("title", "").lower()
        section = chunk.get("section", "").lower()
        content = chunk.get("content", "").lower()

        is_legal_page = ("privacy" in url or "terms" in url or "privacy" in title or "terms" in title)

        if is_legal_page and intent not in ["privacy_policy", "terms_conditions"]:
            return False

        if intent == "ielts_classes":
            return any(w in content or w in section for w in ["ielts", "coaching", "faculty", "education", "consultant"])

        if intent == "services":
            return any(w in content or w in section for w in ["service", "counseling", "counselling", "education", "consultant", "visa"])

        if intent == "visa_services":
            return any(w in content or w in section for w in ["visa", "study", "abroad", "education", "consultant", "university", "college", "counseling"])

        return not is_legal_page

async def main():
    await connect_to_mongodb()
    db = await get_database()
    chunks = await db.website_chunks.find({"active": True}).to_list(1000)

    test_cases = [
        ("TEST 1", "Are you providing IELTS classes?", "ielts_classes"),
        ("TEST 3", "services", "services"),
        ("TEST 4", "help me to get the student visa", "visa_services"),
        ("TEST 5", "I need help with a study visa", "visa_services"),
        ("TEST 6", "I want to study abroad", "visa_services"),
        ("TEST 7", "What countries do you provide student visa services for?", "visa_services"),
        ("TEST 8", "What is the mass of Saturn's moons?", "unknown"),
    ]

    for label, q, intent in test_cases:
        print("="*70)
        print(f"{label} | QUERY: '{q}' | INTENT: {intent}")
        tokens = WebsiteQueryNormalizer.normalize_query(q)
        scored = []
        for c in chunks:
            s = RelevanceScorerTest.score_chunk(c, q, tokens, intent=intent)
            if s > 0 and ValidatorTest.validate_chunk(c, intent):
                scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        print(f"Scored & Valid Chunks: {len(scored)}")
        for score, c in scored[:3]:
            print(f"  Score: {score:.1f} | Sec: '{c.get('section')}' | URL: {c.get('canonical_url')}")
            print(f"    Snippet: {c.get('content')[:140].strip()}...")
        if not scored:
            print("  NO VALID CHUNKS FOUND -> Triggering Fallback!")

if __name__ == "__main__":
    asyncio.run(main())
