"""
Precious AI — Level 4 Domain Knowledge Evaluator Runner

Evaluates visa terminology, definitions, differences, explanations, common domain questions,
and multi-turn domain context inheritance.
"""

import json
import time
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.evaluation.runners.runner_base import EvaluationRunnerBase
from app.evaluation.metrics.eval_metrics import EvaluationMetrics


class DomainKnowledgeEvaluator(EvaluationRunnerBase):
    """
    Evaluator for Domain Knowledge.
    """

    def __init__(self, dataset_path: Optional[Path] = None):
        super().__init__(db_name="precious_ai_eval_domain")
        self.dataset_path = dataset_path or (
            Path(__file__).parent.parent / "datasets" / "domain" / "domain_golden.json"
        )

    async def run(self) -> Dict[str, Any]:
        """
        Executes domain knowledge evaluation suite.
        """
        await self.setup()

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset_data = json.load(f)

        test_cases = dataset_data.get("test_cases", [])
        results = []

        session = await self.session_service.create_session("Domain Eval Session")
        session_id = session["session_id"]

        start_time = time.perf_counter()

        for test in test_cases:
            test_id = test["id"]

            if "prompt" in test:
                prompt = test["prompt"]
                keywords = test.get("keywords", [])
                t0 = time.perf_counter()
                try:
                    res = await self.engine.handle_message(session_id, prompt)
                    latency = time.perf_counter() - t0

                    response_text = res.get("response", "")
                    is_valid = EvaluationMetrics.is_valid_response(response_text)
                    grounding_score = EvaluationMetrics.evaluate_grounding(response_text, keywords)
                    eos_check = EvaluationMetrics.check_eos_termination(response_text)

                    passed = is_valid and grounding_score > 0.0 and eos_check["clean_termination"]

                    results.append({
                        "id": test_id,
                        "prompt": prompt,
                        "response": response_text,
                        "passed": passed,
                        "grounding_score": grounding_score,
                        "latency": latency,
                        "failure_category": None if passed else "DOMAIN"
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
                        "failure_category": self.classify_failure(str(e), "DOMAIN")
                    })
            elif "turns" in test:
                # Multi-turn domain evaluation
                turn_results = []
                multi_session = await self.session_service.create_session(f"Domain Multi {test_id}")
                ms_id = multi_session["session_id"]
                all_passed = True

                for turn in test["turns"]:
                    user_text = turn["user"]
                    t0 = time.perf_counter()
                    try:
                        res = await self.engine.handle_message(ms_id, user_text)
                        lat = time.perf_counter() - t0
                        resp_text = res.get("response", "")
                        valid = EvaluationMetrics.is_valid_response(resp_text)

                        turn_passed = valid
                        if not turn_passed:
                            all_passed = False

                        turn_results.append({
                            "user": user_text,
                            "response": resp_text,
                            "passed": turn_passed,
                            "latency": lat
                        })
                    except Exception as e:
                        all_passed = False
                        turn_results.append({
                            "user": user_text,
                            "error": str(e),
                            "passed": False
                        })

                results.append({
                    "id": test_id,
                    "category": test.get("category"),
                    "passed": all_passed,
                    "turns": turn_results,
                    "failure_category": None if all_passed else "DOMAIN"
                })

        total_time = time.perf_counter() - start_time
        await self.teardown()

        summary = EvaluationMetrics.compute_summary_scorecard(results)
        summary["total_duration_sec"] = total_time
        summary["results"] = results
        return summary


def main():
    evaluator = DomainKnowledgeEvaluator()
    summary = asyncio.run(evaluator.run())
    print("\n========================================")
    print("DOMAIN KNOWLEDGE EVALUATION SUMMARY")
    print("========================================")
    print(f"Total Tests : {summary['total_tests']}")
    print(f"Passed      : {summary['passed']}")
    print(f"Failed      : {summary['failed']}")
    print(f"Pass Rate   : {summary['pass_rate']*100:.1f}%")
    print(f"Duration    : {summary['total_duration_sec']:.2f}s")
    print("========================================\n")


if __name__ == "__main__":
    main()
