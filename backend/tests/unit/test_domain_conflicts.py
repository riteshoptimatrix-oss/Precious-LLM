import pytest
import json
from pathlib import Path

from app.domain.schema import DomainRecord, DomainProvenance
from app.domain.conflicts import DomainConflictChecker

def test_domain_conflict_checker_no_conflicts():
    checker = DomainConflictChecker()
    prov = DomainProvenance(source_id="1", source_name="m")
    records = [
        DomainRecord(record_id="r1", type="terminology", term="F1", definition="US academic student visa.", provenance=prov),
        DomainRecord(record_id="r2", type="terminology", term="M1", definition="US vocational student visa.", provenance=prov)
    ]
    conflicts = checker.scan_conflicts(records)
    assert len(conflicts) == 0

def test_domain_conflict_checker_detects_conflict(tmp_path):
    output_report = tmp_path / "conflicts.json"
    checker = DomainConflictChecker()
    
    prov1 = DomainProvenance(source_id="doc1", source_name="manual_v1")
    prov2 = DomainProvenance(source_id="doc2", source_name="manual_v2")
    
    records = [
        DomainRecord(
            record_id="rec_1",
            type="terminology",
            category="study_visa",
            term="F1 Visa",
            definition="Academic visa requiring Form I-20.",
            provenance=prov1
        ),
        DomainRecord(
            record_id="rec_2",
            type="terminology",
            category="study_visa",
            term="F1 Visa",
            definition="Tourist visa not requiring Form I-20.",
            provenance=prov2
        )
    ]
    
    conflicts = checker.scan_conflicts(records)
    assert len(conflicts) == 1
    assert conflicts[0]["term"] == "F1 Visa"
    
    checker.save_conflict_report(conflicts, output_report)
    assert output_report.exists()
