"""
Precious Edu LLM — CLI Prepare Domain Dataset

Generates seed domain training records, applies validation, deduplication,
conflict checking, and splits into training and evaluation sets with manifest creation.

Usage:
    python -m scripts.prepare_domain_dataset
"""

import sys
from pathlib import Path

from app.domain.dataset import DomainDatasetPipeline


def main():
    print("=" * 60)
    print("PRECIOUS AI — DOMAIN DATASET PREPARATION PIPELINE")
    print("=" * 60)

    pipeline = DomainDatasetPipeline()
    manifest = pipeline.process_and_save()

    print(f"Total Raw Records: {manifest['total_raw_records']}")
    print(f"Valid Records: {manifest['valid_records']}")
    print(f"Quarantined Records: {manifest['quarantined_records']}")
    print(f"Conflicts Detected: {manifest['conflicts_detected']}")
    print(f"Train Records: {manifest['train_records']} -> {manifest['train_file']}")
    print(f"Validation Records: {manifest['val_records']} -> {manifest['val_file']}")
    print(f"Golden Benchmark File: {manifest['golden_file']}")
    print("-" * 60)
    print("SUCCESS: Domain dataset prepared successfully.")


if __name__ == "__main__":
    main()
