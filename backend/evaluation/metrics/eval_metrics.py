"""
Precious AI — Evaluation Metrics Engine

Calculates measurable metrics for response quality, repetition, EOS termination,
context grounding, hallucination rate, refusal handling, and performance.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class EvaluationMetrics:
    """
    Computes deterministic metrics on generated model responses.
    """

    @staticmethod
    def is_valid_response(response: str) -> bool:
        """Returns True if response is a non-empty string."""
        if not response or not isinstance(response, str):
            return False
        return len(response.strip()) > 0

    @staticmethod
    def detect_repetition(response: str) -> Dict[str, Any]:
        """
        Detects word-level, line-level, and n-gram repetition loops.
        """
        if not response:
            return {"has_repetition": False, "score": 0.0, "type": None}

        lines = [line.strip() for line in response.splitlines() if line.strip()]

        # Line-level duplicate check
        if len(lines) > 1 and len(set(lines)) < len(lines):
            return {"has_repetition": True, "score": 1.0, "type": "line_repetition"}

        # Token n-gram repetition check (3-grams)
        words = re.findall(r'\b\w+\b', response.lower())
        if len(words) >= 6:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            unique_trigrams = set(trigrams)
            ratio = 1.0 - (len(unique_trigrams) / len(trigrams))
            if ratio > 0.4:
                return {"has_repetition": True, "score": float(ratio), "type": "ngram_repetition"}

        return {"has_repetition": False, "score": 0.0, "type": None}

    @staticmethod
    def check_eos_termination(response: str) -> Dict[str, Any]:
        """
        Verifies that generated output terminates cleanly without leaking control tokens.
        """
        control_tokens = ["<eos>", "<pad>", "<bos>", "<user>", "<system>", "<assistant>"]
        leaked = [tok for tok in control_tokens if tok in response]

        has_control_leak = len(leaked) > 0
        return {
            "clean_termination": not has_control_leak,
            "leaked_tokens": leaked
        }

    @staticmethod
    def evaluate_grounding(response: str, expected_keywords: List[str]) -> float:
        """
        Calculates ratio of expected grounding keywords present in response.
        """
        if not expected_keywords:
            return 1.0
        response_lower = response.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
        return float(matched / len(expected_keywords))

    @staticmethod
    def detect_hallucination(
        response: str,
        forbidden_keywords: List[str],
        is_unknown_query: bool = False
    ) -> float:
        """
        Calculates hallucination score (0.0 = clean, 1.0 = heavy hallucination).
        If is_unknown_query is True, inventing confident status/facts counts as hallucination.
        """
        response_lower = response.lower()

        # Check explicit forbidden keywords
        for forbidden in forbidden_keywords:
            if forbidden.lower() in response_lower:
                return 1.0

        # For unknown queries, check if response confidently asserts project status without refusal
        if is_unknown_query:
            invented_indicators = ["status is in progress", "status is completed", "managed by", "project xyz is"]
            refusal_indicators = ["not found", "no matching", "no project", "does not exist", "don't have", "cannot find"]

            has_invented = any(ind in response_lower for ind in invented_indicators)
            has_refusal = any(ref in response_lower for ref in refusal_indicators)

            if has_invented and not has_refusal:
                return 1.0

        return 0.0

    @staticmethod
    def compute_summary_scorecard(eval_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates individual test case results into category-level scores.
        """
        total = len(eval_results)
        if total == 0:
            return {"total_tests": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}

        passed = sum(1 for r in eval_results if r.get("passed", False))
        failed = total - passed

        return {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": float(passed / total)
        }
