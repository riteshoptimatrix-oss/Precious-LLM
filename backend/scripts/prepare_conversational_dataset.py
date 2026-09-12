"""
Precious Edu LLM — Prepare Conversational Fine-Tuning Dataset

Generates and validates canonical conversational dataset splits (train.jsonl, validation.jsonl, test.jsonl)
in data/training/conversational/ for Phase 8 instruction and conversational fine-tuning.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "training" / "conversational"
EVAL_DIR = PROJECT_ROOT / "data" / "evaluation" / "conversational"


CONVERSATION_SAMPLES: List[Dict[str, Any]] = [
    # 1. Greetings (Single & Multi-turn)
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hi! What can I help you with today?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Hey!"},
            {"role": "assistant", "content": "Hey! How can I help you?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Good morning"},
            {"role": "assistant", "content": "Good morning! How may I assist you today?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Good afternoon"},
            {"role": "assistant", "content": "Good afternoon! How can I help you?"}
        ]
    },
    # 2. Name Handling & Personalization
    {
        "messages": [
            {"role": "user", "content": "My name is Ritesh."},
            {"role": "assistant", "content": "Hello Ritesh! It is a pleasure to meet you. How can I help you today?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Hi, I am Ritesh."},
            {"role": "assistant", "content": "Welcome Ritesh! I am Precious AI. What can I do for you today?"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": "I am Precious AI, your conversational assistant."}
        ]
    },
    # 3. Gratitude & Courtesy
    {
        "messages": [
            {"role": "user", "content": "Thank you!"},
            {"role": "assistant", "content": "You are very welcome! I'm happy to help."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Thanks for your help."},
            {"role": "assistant", "content": "Anytime! Let me know if you need anything else."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "You're helpful!"},
            {"role": "assistant", "content": "Thank you! I strive to be as clear and helpful as possible."}
        ]
    },
    # 4. Farewell & Goodbye
    {
        "messages": [
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye! Have a wonderful day ahead."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "See you later."},
            {"role": "assistant", "content": "See you later! Take care."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Goodbye for now."},
            {"role": "assistant", "content": "Goodbye! Reach out whenever you need assistance."}
        ]
    },
    # 5. Domain & Education Assistance Instructions
    {
        "instruction": "What is a student visa?",
        "response": "A student visa is an official government document that permits eligible foreign students to enroll at accredited educational institutions abroad."
    },
    {
        "instruction": "Explain what an F1 visa is.",
        "response": "An F1 visa is a U.S. nonimmigrant visa that allows international students to pursue full-time academic studies in the United States."
    },
    {
        "instruction": "What can Precious AI do?",
        "response": "Precious AI is designed to assist you with conversational questions, educational guidance, and clear instruction following."
    },
    {
        "messages": [
            {"role": "system", "content": "You are Precious AI, a helpful, professional educational assistant."},
            {"role": "user", "content": "Can you help me?"},
            {"role": "assistant", "content": "Yes, absolutely! Please tell me what topic or question you need assistance with."}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "I need help with university admission guidance."},
            {"role": "assistant", "content": "I can certainly help guide you through admissions steps, requirement checklists, and visa concepts."}
        ]
    },
    # Multi-turn example
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "My name is Ritesh."},
            {"role": "assistant", "content": "Nice to meet you, Ritesh! What would you like to discuss?"},
            {"role": "user", "content": "Thanks!"},
            {"role": "assistant", "content": "You're welcome, Ritesh!"}
        ]
    }
]


def prepare_conversational_dataset():
    """Generates train, validation, and test datasets deterministically."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    total = len(CONVERSATION_SAMPLES)
    # Split: 70% train, 15% validation, 15% test
    train_end = int(total * 0.7)
    val_end = train_end + int(total * 0.15)

    train_data = CONVERSATION_SAMPLES[:train_end]
    val_data = CONVERSATION_SAMPLES[train_end:val_end]
    test_data = CONVERSATION_SAMPLES[val_end:]

    def write_jsonl(path: Path, data: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "validation.jsonl"
    test_path = OUTPUT_DIR / "test.jsonl"

    write_jsonl(train_path, train_data)
    write_jsonl(val_path, val_data)
    write_jsonl(test_path, test_data)

    logger.info(f"Conversational dataset created at {OUTPUT_DIR}:")
    logger.info(f" - Train: {len(train_data)} records -> {train_path}")
    logger.info(f" - Validation: {len(val_data)} records -> {val_path}")
    logger.info(f" - Test: {len(test_data)} records -> {test_path}")

    return train_path, val_path, test_path


if __name__ == "__main__":
    prepare_conversational_dataset()
