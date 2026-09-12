"""
Precious Edu LLM — Dataset Pipeline Script

Executes the 13-stage dataset engineering pipeline:
Raw Data Ingestion -> Normalization -> Validation -> Quarantine -> Deduplication ->
Quality Filtering -> Statistics -> Deterministic Splitting -> JSONL Writing -> Manifest Generation.
"""

import sys
import logging
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dataset.pipeline import DatasetPipeline
from app.dataset.config import get_dataset_config


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    print("==================================================")
    print("Precious Edu LLM — Preparing Training Dataset")
    print("==================================================")

    config = get_dataset_config()
    pipeline = DatasetPipeline(config=config)
    results = pipeline.run()

    print("\n--------------------------------------------------")
    print("Pipeline Execution Results:")
    print("--------------------------------------------------")
    print(f"Raw Records Read:      {results['records_read']}")
    print(f"Valid Processed:       {results['valid_records']}")
    print(f"Train Split:           {results['train_records']}")
    print(f"Validation Split:      {results['validation_records']}")
    print(f"Test Split:            {results['test_records']}")
    print(f"Rejection Pass Rate:   {results['quality_report']['pass_rate_pct']}%")
    print("--------------------------------------------------")
    print(f"Manifest & Statistics written to: {config.METADATA_DIR}")
    print(f"Training outputs written to:      {config.TRAINING_DIR}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
