# Precious Edu LLM — Data Architecture

This directory contains raw, intermediate, processed, training, evaluation, and metadata assets for training the custom LLM.

## Directory Structure

```text
data/
├── raw/                      # Immutable source datasets
│   ├── text/                 # Plain text documents (.txt, .json)
│   ├── conversations/        # Dialogue/chat datasets (.json, .jsonl)
│   └── external/             # Third-party datasets with tracked provenance
│
├── intermediate/             # Pipeline scratch & debugging outputs
│   └── quarantine/           # Rejected/malformed records with error reasons
│
├── cleaned/                  # Normalized & cleaned records
│
├── processed/                # Fully validated & deduplicated records
│
├── training/                 # Final deterministic dataset splits (tokenizer-ready)
│   ├── train.jsonl           # Training split (90% default)
│   ├── validation.jsonl      # Validation split (5% default)
│   └── test.jsonl            # Test split (5% default)
│
├── evaluation/               # Isolated evaluation datasets
│   └── golden_eval.jsonl     # Golden benchmark test set (isolated from training)
│
└── metadata/                 # Dataset manifests, checksums & reports
    ├── dataset_manifest.json # Complete dataset version & source provenance
    ├── statistics.json       # Record, turn, character, and language stats
    └── quality_report.json   # Filtering & quarantine metrics
```

## Immutability Rules
- **DO NOT** edit or modify files in `data/raw/` in place.
- **DO NOT** manually modify files in `data/training/` or `data/metadata/`. Always re-run `python -m scripts.prepare_dataset`.
- **DO NOT** train models on `data/evaluation/` sets.
