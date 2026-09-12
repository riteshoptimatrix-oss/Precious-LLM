# Precious Edu LLM - Run & Management Guide

This document provides instructions on how to train the models, evaluate their performance, and run the backend server. All commands should be run from the `backend/` directory.

## 1. Running the Backend Server

Start the FastAPI backend server using Uvicorn:

```bash
python run.py
# Or explicitly:
python run.py server
```

## 2. Training the Model (Unified run.py CLI)

You can train the model directly using `run.py` from either the project root or the `backend/` directory:

```bash
# Complete end-to-end training pipeline:
python run.py train-all

# Or run individual stages:
# Step 1: Prepare Dataset (train/val/test splits)
python run.py prepare-data

# Step 2: Train BPE Tokenizer
python run.py train-tokenizer

# Step 3: Tokenize Training Dataset
python run.py tokenize-data

# Step 4: Train Base Transformer Model
python run.py train-base
python run.py train-base --epochs 10 --dry-run

# Step 5: Domain Adaptation Training
python run.py train-domain
python run.py train-domain --lr 0.0001 --epochs 5

# Step 6: Conversational Fine-Tuning
python run.py finetune
python run.py finetune --epochs 5 --learning_rate 0.00005
```

## 3. Managing Knowledge & Importer

```bash
# Structured Q&A (MongoDB structured_qa)
python run.py import-qa
python run.py import-qa --stats-only

# Website Knowledge (MongoDB website_chunks)
python run.py crawl-website
python run.py crawl-website --dry-run
python run.py refresh-website
```

## 4. Evaluation

We have several evaluation scripts to test different aspects of the LLM pipeline.

**Evaluate Base Model:**
```bash
python -m scripts.evaluate_model
```

**Evaluate Fine-tuned Model:**
```bash
python -m scripts.evaluate_finetuned_model
```

**Evaluate Domain Knowledge:**
```bash
python -m scripts.evaluate_domain
```

**Evaluate Website Knowledge Retrieval:**
```bash
python -m scripts.evaluate_website_knowledge
```

**Evaluate Safety & Guardrails:**
```bash
python -m scripts.evaluate_safety
```

**End-to-End Chat Evaluation:**
```bash
python -m scripts.evaluate_e2e
```

## 5. Helpful Inspection Scripts

To inspect data and tokenization, you can use:
```bash
python -m scripts.inspect_dataset
python -m scripts.inspect_domain_dataset
python -m scripts.inspect_tokenizer
python -m scripts.inspect_model
python -m scripts.inspect_conversations
```
