"""
Precious AI — Phase 12 Integrated Evaluation & Safety Pytest Suite

Executes evaluation layer suites under pytest:
- General conversation
- Domain knowledge
- Project knowledge
- Safety, security & adversarial
- End-to-end session isolation
"""

import pytest
from app.evaluation.runners.evaluate_general import GeneralConversationEvaluator
from app.evaluation.runners.evaluate_domain import DomainKnowledgeEvaluator
from app.evaluation.runners.evaluate_project import ProjectKnowledgeEvaluator
from app.evaluation.runners.evaluate_safety import SafetySecurityEvaluator
from app.evaluation.runners.evaluate_e2e import EndToEndEvaluator
from app.evaluation.reports.generate_evaluation_report import EvaluationReportGenerator


@pytest.mark.asyncio
async def test_level3_general_conversation_evaluation():
    evaluator = GeneralConversationEvaluator()
    summary = await evaluator.run()
    assert summary["total_tests"] > 0
    assert summary["pass_rate"] >= 0.8


@pytest.mark.asyncio
async def test_level4_domain_knowledge_evaluation():
    evaluator = DomainKnowledgeEvaluator()
    summary = await evaluator.run()
    assert summary["total_tests"] > 0
    assert summary["pass_rate"] >= 0.8


@pytest.mark.asyncio
async def test_level4_project_knowledge_evaluation():
    evaluator = ProjectKnowledgeEvaluator()
    summary = await evaluator.run()
    assert summary["total_tests"] > 0
    assert summary["pass_rate"] >= 0.8


@pytest.mark.asyncio
async def test_level4_safety_security_evaluation():
    evaluator = SafetySecurityEvaluator()
    summary = await evaluator.run()
    assert summary["total_tests"] > 0
    assert summary["pass_rate"] >= 0.8


@pytest.mark.asyncio
async def test_level5_end_to_end_session_isolation_evaluation():
    evaluator = EndToEndEvaluator()
    summary = await evaluator.run()
    assert summary["total_tests"] > 0
    assert summary["pass_rate"] >= 0.8


@pytest.mark.asyncio
async def test_evaluation_report_generation():
    generator = EvaluationReportGenerator()
    report_dir = await generator.generate()
    assert report_dir.exists()
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "report.md").exists()
