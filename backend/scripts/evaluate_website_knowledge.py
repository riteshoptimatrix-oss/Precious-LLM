"""
Precious AI — CLI: Evaluate Website Knowledge Retrieval

Runs a deterministic Q&A test suite against the active website knowledge index.
Measures: precision@K (correct source retrieved), source grounding
(answer contains website content), and no-match handling (empty results
for irrelevant queries should not hallucinate website content).

Usage:
    python -m scripts.evaluate_website_knowledge
    python -m scripts.evaluate_website_knowledge --top-k 5 --min-score 3.0
"""

import argparse
import asyncio
import logging
import sys
import os
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.mongodb import connect_to_mongodb, get_database, close_mongodb_connection
from app.website.services.website_knowledge_service import WebsiteKnowledgeService

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate_website_knowledge")

# ---------------------------------------------------------------------------
# Evaluation Test Cases
# Each case: query, expected_keywords (at least one should appear in retrieved content),
#            should_return_results (True/False for no-match tests)
# ---------------------------------------------------------------------------
EVAL_CASES: List[Dict[str, Any]] = [
    # Website service queries
    {"query": "study visa services", "expected_keywords": ["visa", "study", "service"], "should_return": True},
    {"query": "IELTS coaching classes", "expected_keywords": ["ielts", "coaching", "training"], "should_return": True},
    {"query": "study in Canada fees", "expected_keywords": ["canada", "fee", "cost", "tuition"], "should_return": True},
    {"query": "study in Australia admission", "expected_keywords": ["australia", "admission", "university"], "should_return": True},
    {"query": "student visa USA", "expected_keywords": ["usa", "visa", "student", "f1"], "should_return": True},
    {"query": "GRE preparation courses", "expected_keywords": ["gre", "test", "prep", "exam"], "should_return": True},
    {"query": "contact Precious Education office", "expected_keywords": ["contact", "office", "address", "phone", "precious", "education"], "should_return": True},
    {"query": "immigration consultancy services", "expected_keywords": ["immigration", "service", "consultant"], "should_return": True},
    # No-match cases — should return no results (no hallucination)
    {"query": "cricket world cup schedule 2024", "expected_keywords": [], "should_return": False},
    {"query": "stock market tips today", "expected_keywords": [], "should_return": False},
    {"query": "pizza recipe italian", "expected_keywords": [], "should_return": False},
]


async def main(args: argparse.Namespace) -> None:
    await connect_to_mongodb()
    db = await get_database()

    try:
        service = WebsiteKnowledgeService(db, top_k=args.top_k, min_score=args.min_score)

        print(f"\n{'=' * 70}")
        print(f"  Website Knowledge Evaluation — top_k={args.top_k}, min_score={args.min_score}")
        print(f"{'=' * 70}")

        total = len(EVAL_CASES)
        passed = 0
        failed_cases = []

        for case in EVAL_CASES:
            query = case["query"]
            expected_kws = case["expected_keywords"]
            should_return = case["should_return"]

            chunks = await service.search(query, top_k=args.top_k)
            returned_results = len(chunks) > 0

            if not should_return:
                # No-match test: should return nothing
                ok = not returned_results
                status = "PASS" if ok else "FAIL (returned results for irrelevant query)"
            else:
                # Positive test: should return results containing expected keywords
                if not returned_results:
                    ok = False
                    status = "FAIL (no results returned)"
                else:
                    combined_content = " ".join(
                        (c.get("content", "") + " " + c.get("title", "") + " " + c.get("section", "")).lower()
                        for c in chunks
                    )
                    kw_hits = [kw for kw in expected_kws if kw in combined_content]
                    ok = len(kw_hits) > 0
                    if ok:
                        status = f"PASS (keywords matched: {kw_hits})"
                    else:
                        status = f"FAIL (no expected keywords found; got {len(chunks)} chunks)"

            if ok:
                passed += 1
            else:
                failed_cases.append({"query": query, "status": status})

            print(f"  [{('PASS' if ok else 'FAIL')}] \"{query}\"")
            print(f"         {status}")

        print(f"\n{'=' * 70}")
        precision = (passed / total) * 100 if total > 0 else 0.0
        print(f"  Results: {passed}/{total} passed  ({precision:.1f}% precision)")

        if failed_cases:
            print(f"\n  Failed Cases ({len(failed_cases)}):")
            for f in failed_cases:
                print(f"    FAIL: \"{f['query']}\" -> {f['status']}")

        print(f"{'=' * 70}\n")

        if passed < total:
            sys.exit(1)
    finally:
        await close_mongodb_connection()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate website knowledge retrieval accuracy."
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-K chunks to retrieve per query (default: 3).")
    parser.add_argument("--min-score", type=float, default=5.0, help="Minimum relevance score threshold (default: 5.0).")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
