import pytest
import json
from pathlib import Path

from app.domain.evaluator import DomainEvaluator
from app.domain_training.config import DomainTrainingConfig

def test_domain_evaluator_initialization():
    config = DomainTrainingConfig()
    evaluator = DomainEvaluator(
        base_model_path=str(config.resolved_base_checkpoint),
        domain_model_path=str(config.resolved_base_checkpoint),
        tokenizer_path=str(config.resolved_tokenizer_dir)
    )
    assert evaluator.base_model_path is not None

def test_domain_evaluator_evaluate_all():
    config = DomainTrainingConfig()
    evaluator = DomainEvaluator(
        base_model_path=str(config.resolved_base_checkpoint),
        domain_model_path=str(config.resolved_base_checkpoint),
        tokenizer_path=str(config.resolved_tokenizer_dir)
    )
    report = evaluator.evaluate_all()
    assert "base_model_path" in report
    assert "domain_model_path" in report
    assert "summary" in report
    assert report["summary"]["general_conversation_retention_pass"] is True
    assert report["summary"]["project_knowledge_isolation_pass"] is True
