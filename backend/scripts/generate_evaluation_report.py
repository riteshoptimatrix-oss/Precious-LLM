"""
Precious AI — CLI Script: Generate Evaluation Report

Usage:
    python -m scripts.generate_evaluation_report
"""

import sys
import asyncio
from app.evaluation.reports.generate_evaluation_report import EvaluationReportGenerator


def main():
    generator = EvaluationReportGenerator()
    report_path = asyncio.run(generator.generate())
    print(f"Report directory created at: {report_path}")


if __name__ == "__main__":
    main()
