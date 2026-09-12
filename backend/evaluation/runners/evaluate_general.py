"""
Precious AI — Level 3 General Conversation Evaluator Runner

Evaluates greetings, name memory recall, thanks, bye, natural conversation flow,
and verifies no hallucinated facts on basic interactions.
"""

import json
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List

from app.evaluation.runners.runner_base import EvaluationRunnerBase
from app.evaluation.metrics.eval_metrics import EvaluationMetrics


class GeneralConversationEvaluator(EvaluationRunnerBase):
    """
    Evaluator for General Conversation.
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        super().__init__(db_name="precious_ai_eval_general")
        self.dataset_path = dataset_path or (
            Path(__file__).parent.parent / "datasets" / "general" / "golden_general.json"
        )

    async def run(self) -> Dict[str, Any]:
        """
        Executes general conversation test suite.
        """
        await self.setup()

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset_data = json.load(f)

        test_cases = dataset_data.get("test_cases", [])
        results = []

        session = await self.session_service.create_session("General Eval Session")
        session_id = session["session_id"]

        start_time = time.perf_counter()

        for test in test_cases:
            test_id = test["id"]
            prompt = test["prompt"]
            forbidden = test.get("forbidden_tokens", [])

            t0 = time.perf_counter()
            try:
                res = await self.engine.handle_message(session_id, prompt)
                latency = time.perf_counter() - t0

                response_text = res.get("response", "")
                is_valid = EvaluationMetrics.is_valid_response(response_text)
                eos_check = EvaluationMetrics.check_eos_termination(response_text)
                rep_check = EvaluationMetrics.detect_repetition(response_text)

                # Check forbidden tokens
                has_forbidden = any(f.lower() in response_text.lower() for f in forbidden)

                passed = is_valid and eos_check["clean_termination"] and not rep_check["has_repetition"] and not has_forbidden

                failure_category = None
                if not passed:
                    if not is_valid or not eos_check["clean_termination"]:
                        failure_category = "GENERATION"
                    elif rep_check["has_repetition"]:
                        failure_category = "GENERATION"
                    elif has_forbidden:
                        failure_category = "HALLUCINATION"

                results.append({
                    "id": test_id,
                    "prompt": prompt,
                    "response": response_text,
                    "passed": passed,
                    "latency": latency,
                    "failure_category": failure_category,
                    "repetition": rep_check,
                    "eos_clean": eos_check["clean_termination"]
                })
            except Exception as e:
                latency = time.perf_counter() - t0
                results.append({
                    "id": test_id,
                    "prompt": prompt,
                    "response": None,
                    "passed": False,
                    "latency": latency,
                    "error": str(e),
                    "failure_category": self.classify_failure(str(e))
                })

        total_time = time.perf_counter() - start_time
        await self.teardown()

        summary = EvaluationMetrics.compute_summary_scorecard(results)
        summary["total_duration_sec"] = total_time
        summary["results"] = results
        return summary


def main():
    evaluator = GeneralConversationEvaluator()
    summary = asyncio.run(evaluator.run())
    print("\n========================================")
    print("GENERAL CONVERSATION EVALUATION SUMMARY")
    print("========================================")
    print(f"Total Tests : {summary['total_tests']}")
    print(f"Passed      : {summary['passed']}")
    print(f"Failed      : {summary['failed']}")
    print(f"Pass Rate   : {summary['pass_rate']*100:.1f}%")
    print(f"Duration    : {summary['total_duration_sec']:.2f}s")
    print("========================================\n")


if __name__ == "__main__":
    main()
