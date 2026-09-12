"""
Precious Edu LLM — CLI Validate Domain Dataset

Runs schema validation, duplicate detection, security secret scanning,
and domain conflict report generation.

Usage:
    python -m scripts.validate_domain_dataset
"""

import json
from pathlib import Path

from app.domain.conflicts import DomainConflictChecker
from app.domain.validator import DomainValidator


def main():
    print("=" * 60)
    print("PRECIOUS AI — DOMAIN DATASET VALIDATION & SECURITY SCAN")
    print("=" * 60)

    train_path = Path("data/domain/training/train.jsonl")
    if not train_path.exists():
        print("ERROR: Training dataset file does not exist. Run prepare_domain_dataset first.")
        return

    raw_records = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    validator = DomainValidator()
    checker = DomainConflictChecker()

    valid, quarantined = validator.validate_batch(raw_records)
    conflicts = checker.scan_conflicts(valid)

    print(f"Inspected Records: {len(raw_records)}")
    print(f"Valid Records: {len(valid)}")
    print(f"Quarantined Items: {len(quarantined)}")
    print(f"Conflicts Detected: {len(conflicts)}")

    if quarantined:
        print("\n--- QUARANTINE REPORT ---")
        for q in quarantined:
            print(f"Record {q.record_id} -> Reason: {q.reason} | Errors: {', '.join(q.errors)}")

    if conflicts:
        print("\n--- CONFLICTS REPORT ---")
        for c in conflicts:
            print(f"Term: {c.get('term')} -> {c.get('description')}")

    if not quarantined and not conflicts:
        print("\nSUCCESS: 0 errors, 0 secrets detected, 0 conflicts.")

    print("=" * 60)


if __name__ == "__main__":
    main()
