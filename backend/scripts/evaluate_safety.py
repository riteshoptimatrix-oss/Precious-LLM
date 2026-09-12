"""
Precious AI — CLI Script: Safety, Security & Adversarial Evaluation

Usage:
    python -m scripts.evaluate_safety
"""

import sys
import asyncio
from app.evaluation.runners.evaluate_safety import SafetySecurityEvaluator


def main():
    evaluator = SafetySecurityEvaluator()
    summary = asyncio.run(evaluator.run())
    print("\n========================================")
    print("SAFETY & ADVERSARIAL EVALUATION SUMMARY")
    print("========================================")
    print(f"Total Tests : {summary['total_tests']}")
    print(f"Passed      : {summary['passed']}")
    print(f"Failed      : {summary['failed']}")
    print(f"Pass Rate   : {summary['pass_rate']*100:.1f}%")
    print(f"Duration    : {summary['total_duration_sec']:.2f}s")
    print("========================================\n")
    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
