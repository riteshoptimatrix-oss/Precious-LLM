import pytest
import json
from pathlib import Path

from app.domain.schema import DomainRecord, DomainProvenance
from app.domain.validator import DomainValidator
from app.domain.dataset import DomainDatasetPipeline

def test_domain_schema_valid():
    prov = DomainProvenance(source_id="src_1", source_name="manual")
    record = DomainRecord(
        record_id="rec_001",
        type="instruction_response",
        instruction="What is an F1 visa?",
        response="An F1 visa is a student visa for academic studies in the USA.",
        provenance=prov
    )
    assert record.type == "instruction_response"
    assert record.provenance.source_id == "src_1"

def test_domain_validator_valid():
    validator = DomainValidator()
    record = {
        "record_id": "rec_001",
        "type": "instruction_response",
        "instruction": "What is an F1 visa?",
        "response": "An F1 visa is for academic studies.",
        "provenance": {"source_id": "src_1", "source_name": "manual"}
    }
    valid_record, quarantined = validator.validate_record(record)
    assert valid_record is not None
    assert quarantined is None
    assert valid_record.record_id == "rec_001"

def test_domain_validator_secret_scanning():
    validator = DomainValidator()
    record = {
        "record_id": "rec_002",
        "type": "instruction_response",
        "instruction": "What is the key?",
        "response": "Here is the key: sk-abcdefghijklmnopqrstuvwxyz123456",
        "provenance": {"source_id": "src_1", "source_name": "manual"}
    }
    valid_record, quarantined = validator.validate_record(record)
    assert valid_record is None
    assert quarantined is not None
    assert any("secret" in e.lower() for e in quarantined.errors)

def test_domain_validator_batch_dedup():
    validator = DomainValidator()
    records = [
        {"record_id": "rec_001", "type": "terminology", "term": "F1", "definition": "Academic visa.", "provenance": {"source_id": "1", "source_name": "m"}},
        {"record_id": "rec_002", "type": "terminology", "term": "F1", "definition": "Academic visa.", "provenance": {"source_id": "2", "source_name": "m"}}
    ]
    valid, quarantined = validator.validate_batch(records)
    assert len(valid) == 1
    assert len(quarantined) == 1
    assert "duplicate" in quarantined[0].reason.lower()

def test_domain_dataset_pipeline(tmp_path):
    pipeline = DomainDatasetPipeline(data_root=tmp_path)
    train_count, val_count, eval_count = pipeline.run_pipeline()
    
    assert train_count > 0
    assert val_count >= 0
    assert eval_count > 0
    assert (tmp_path / "training" / "train.jsonl").exists()
    assert (tmp_path / "evaluation" / "golden.jsonl").exists()
