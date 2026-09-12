"""
Precious Edu LLM — Tokenize Dataset Script

Tokenizes Phase 4 dataset splits (train.jsonl, validation.jsonl, test.jsonl)
using the frozen trained tokenizer and saves output token IDs to data/processed/tokenized/.
"""

import sys
import json
import logging
from pathlib import Path

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dataset.config import get_dataset_config
from app.tokenizer.config import get_tokenizer_config
from app.tokenizer.tokenizer import Tokenizer
from app.tokenizer.formatter import ConversationFormatter
from app.dataset.models import DatasetRecord, RecordType, MessageRecord


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )

    print("==================================================")
    print("Precious AI — Tokenizing Dataset Splits")
    print("==================================================")

    data_config = get_dataset_config()
    tok_config = get_tokenizer_config()

    if not tok_config.ARTIFACTS_DIR.exists():
        print("Error: Trained tokenizer not found.")
        print("Please run 'python -m scripts.train_tokenizer' first.")
        sys.exit(1)

    tokenizer = Tokenizer.load(tok_config.ARTIFACTS_DIR, config=tok_config)
    formatter = ConversationFormatter(tok_config)

    output_tokenized_dir = data_config.PROCESSED_DIR / "tokenized"
    output_tokenized_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "validation", "test"]

    for split_name in splits:
        input_file = data_config.TRAINING_DIR / f"{split_name}.jsonl"
        output_file = output_tokenized_dir / f"{split_name}_tokenized.jsonl"

        if not input_file.exists():
            print(f"Skipping {split_name} (file not found: {input_file})")
            continue

        tokenized_count = 0
        with open(input_file, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
            for line_idx, line in enumerate(f_in):
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    formatted_text = ""
                    if "messages" in data and isinstance(data["messages"], list):
                        msgs = [MessageRecord(role=m.get("role", ""), content=m.get("content", "")) for m in data["messages"]]
                        rec = DatasetRecord(record_id=f"rec_{line_idx}", source_id=split_name, type=RecordType.CONVERSATION, messages=msgs)
                        formatted_text = formatter.format_record(rec)
                    elif "text" in data and isinstance(data["text"], str):
                        rec = DatasetRecord(record_id=f"rec_{line_idx}", source_id=split_name, type=RecordType.PLAIN_TEXT, text=data["text"])
                        formatted_text = formatter.format_record(rec)

                    if formatted_text:
                        input_ids = tokenizer.encode(formatted_text, add_bos=True, add_eos=True)
                        tokenized_record = {
                            "input_ids": input_ids,
                            "num_tokens": len(input_ids),
                            "metadata": {
                                "split": split_name,
                                "tokenizer_version": tok_config.TOKENIZER_VERSION,
                                "dataset_version": data_config.DATASET_VERSION
                            }
                        }
                        f_out.write(json.dumps(tokenized_record, ensure_ascii=False) + "\n")
                        tokenized_count += 1

                except Exception as err:
                    print(f"Error tokenizing line {line_idx} in {split_name}: {err}")

        print(f"Tokenized {split_name}: {tokenized_count} records -> {output_file.name}")

    print("==================================================\n")


if __name__ == "__main__":
    main()
