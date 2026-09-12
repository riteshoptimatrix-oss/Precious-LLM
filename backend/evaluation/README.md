# Precious AI — Evaluation, Safety & Quality Validation Framework (Phase 12)

This directory contains the formal, production-readiness evaluation framework for Precious AI across 5 evaluation levels:
1. **Level 1 — Unit Evaluation**: Component unit checks (tokenizer, attention, embeddings, memory manager, knowledge router).
2. **Level 2 — Model Evaluation**: Custom LLM model architecture, loading, checkpoint validity, and deterministic inference.
3. **Level 3 — Conversation Evaluation**: Multi-turn conversation context, memory extraction, session isolation.
4. **Level 4 — Knowledge Evaluation**: Excel project knowledge ingestion, dynamic status update, source traceability, missing field refusal, unknown project detection, and multiple match disambiguation.
5. **Level 5 — Full End-to-End Evaluation**: HTTP FastAPI endpoint, session flow, MongoDB persistence, Knowledge Engine, Custom LLM.

## Directory Layout
- `datasets/`: Golden and specialized evaluation datasets (`general/`, `conversational/`, `domain/`, `project/`, `safety/`, `adversarial/`, `regression/`).
- `runners/`: Evaluator runner modules (`evaluate_general`, `evaluate_domain`, `evaluate_project`, `evaluate_safety`, `evaluate_e2e`, `evaluate_all`).
- `metrics/`: Deterministic quality, hallucination, repetition, grounding, latency, and resource footprint metrics.
- `reports/`: Generators for JSON/Markdown audit reports saved in `artifacts/evaluation/run-YYYYMMDD-HHMMSS/`.
- `fixtures/`: Mock databases, test workbooks, and manual review sample datasets.

## CLI Usage
Run evaluation modules via python:
```bash
python -m scripts.evaluate_general
python -m scripts.evaluate_domain
python -m scripts.evaluate_project
python -m scripts.evaluate_safety
python -m scripts.evaluate_e2e
python -m scripts.generate_evaluation_report
```
