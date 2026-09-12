# Precious Edu LLM

A domain-specific conversational AI chatbot built entirely from scratch — no external AI APIs, no pretrained models, no pretrained tokenizers.

## Architecture

- **Frontend**: PHP + JavaScript + CSS (ChatGPT-style interface)
- **Backend**: Python + FastAPI + Uvicorn
- **Database**: MongoDB (local)
- **ML**: PyTorch (custom decoder-only Transformer, ~10M parameters)
- **Tokenizer**: Custom BPE tokenizer trained on our corpus

## Project Structure

```
├── frontend/          # PHP/JS/CSS chat interface
├── backend/           # FastAPI backend application
│   ├── app/           # Main application package
│   │   ├── api/       # Routes, schemas, middleware
│   │   ├── services/  # Business logic
│   │   ├── conversation/  # Context & memory management
│   │   ├── knowledge/ # Knowledge retrieval engine
│   │   ├── ml/        # Model, tokenizer, inference, training
│   │   └── db/        # MongoDB client & repositories
│   └── tests/         # Unit, integration, evaluation tests
├── data/              # Training data & datasets
├── checkpoints/       # Model & tokenizer checkpoints
├── configs/           # YAML configuration files
├── scripts/           # Training & utility scripts
└── docs/              # Documentation
```

## Quick Start

```bash
# 1. Set up Python environment
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Start MongoDB locally

# 3. Configure environment
copy .env.example .env

# 4. Run the backend
python run.py
```

## Phase 4 — Dataset Engineering & Training Data Pipeline

Phase 4 establishes the reproducible training-data pipeline for the custom LLM:
- **Raw Data Immutability**: Raw data stored in `data/raw/` is never modified in place.
- **Canonical Representation**: Ingests `.txt`, `.json`, `.jsonl` raw sources into canonical `DatasetRecord` objects.
- **Conservative Normalization**: Normalizes line endings, whitespace, and null bytes while preserving punctuation, capitalization, accents, emojis, and full Unicode scripts.
- **Validation & Quarantine**: Structural and role validation (`user`, `assistant`, `system`). Invalid/malformed records are stored in `data/intermediate/quarantine/`.
- **Deduplication & Quality Filtering**: SHA-256 exact content deduplication and quality filters (empty content, length limits, pathological repetitions).
- **Deterministic Splitting**: Deterministically splits dataset into `train.jsonl` (90%), `validation.jsonl` (5%), and `test.jsonl` (5%) using `SPLIT_SEED=42` with data leakage prevention.
- **Manifests & Checksums**: Generates `dataset_manifest.json`, `statistics.json`, `quality_report.json`, and SHA-256 checksums in `data/metadata/`.

### Dataset Pipeline Commands:
```bash
# Run 13-stage dataset pipeline
python -m scripts.prepare_dataset

# Inspect dataset manifest, statistics, and preview samples
python -m scripts.inspect_dataset --samples 5
```

## Status

✅ **Phase 1 Complete** — FastAPI Foundation & Infrastructure  
✅ **Phase 2 Complete** — PHP Frontend & API Integration  
✅ **Phase 3 Complete** — Conversation Engine, Context Management & Memory  
✅ **Phase 4 Complete** — Dataset Engineering & Training Data Pipeline  

See `docs/architecture.md` for full technical specifications.