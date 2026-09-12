"""
Precious Edu LLM — Domain Dataset Pipeline

Loads, validates, deduplicates, scans for conflicts, and splits domain datasets
into train, validation, and evaluation subsets with provenance manifests.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.domain.conflicts import DomainConflictChecker
from app.domain.schema import DomainExampleType, DomainProvenance, DomainRecord
from app.domain.taxonomy import DomainTaxonomy
from app.domain.validator import DomainValidator

logger = logging.getLogger(__name__)


class DomainDatasetPipeline:
    """
    Pipeline orchestrating domain dataset preparation, validation, and splitting.
    """

    def __init__(self, data_root: str | Path = "data/domain", data_dir: Optional[str | Path] = None):
        self.data_root = Path(data_dir if data_dir is not None else data_root)
        self.validator = DomainValidator()
        self.conflict_checker = DomainConflictChecker()

    def generate_seed_records(self) -> List[Dict[str, Any]]:
        """Generates representative domain training seed records across all 7 types."""
        prov = {
            "source_id": "seed_001",
            "source_name": "precious_edu_domain_manual",
            "source_type": "official_guidelines",
            "version": "1.0.0",
            "language": "en"
        }

        records = [
            # Type 1: Domain knowledge explanation
            {
                "record_id": "dom_001",
                "type": DomainExampleType.INSTRUCTION_RESPONSE.value,
                "category": "study_visa",
                "instruction": "Explain the difference between F1 and M1 student visas.",
                "response": "An F1 visa is issued for full-time academic studies at an accredited college, university, or language training program. An M1 visa is designated for non-academic, vocational, or technical training programs. Both require full-time enrollment and an issued Form I-20.",
                "provenance": prov
            },
            # Type 2: Terminology
            {
                "record_id": "dom_002",
                "type": DomainExampleType.TERMINOLOGY.value,
                "category": "study_visa",
                "term": "Form I-20",
                "definition": "A Form I-20 is a Certificate of Eligibility for Nonimmigrant Student Status issued by SEVP-certified schools in the USA, necessary to apply for an F1 or M1 visa.",
                "provenance": prov
            },
            # Type 3: Service explanation
            {
                "record_id": "dom_003",
                "type": DomainExampleType.SERVICE.value,
                "category": "study_visa",
                "service_name": "Study Visa Guidance",
                "response": "Precious Education offers comprehensive study visa guidance, including university application assistance, Form I-20 verification, financial document preparation, and visa interview preparation.",
                "provenance": prov
            },
            # Type 4: Domain conversation
            {
                "record_id": "dom_004",
                "type": DomainExampleType.CONVERSATION.value,
                "category": "study_visa",
                "messages": [
                    {"role": "user", "content": "I want to apply for a student visa to study in the USA."},
                    {"role": "assistant", "content": "That is wonderful! To study in the USA, you will typically need an F1 visa after obtaining an acceptance letter and Form I-20 from a SEVP-certified university. Which program or university are you targeting?"}
                ],
                "provenance": prov
            },
            # Type 5: Clarification behavior
            {
                "record_id": "dom_005",
                "type": DomainExampleType.CLARIFICATION.value,
                "category": "general",
                "instruction": "I need a visa.",
                "response": "I would be happy to help! Could you please share which country you plan to visit and whether your primary purpose is study, work, tourism, or permanent residency?",
                "provenance": prov
            },
            # Type 6: Ambiguous question handling
            {
                "record_id": "dom_006",
                "type": DomainExampleType.AMBIGUITY.value,
                "category": "study_visa",
                "instruction": "How long will my application take?",
                "response": "Processing times vary depending on the target country, embassy location, and visa category. Student visa processing generally ranges from 2 to 8 weeks. Providing complete and accurate documentation helps avoid unnecessary delays.",
                "provenance": prov
            },
            # Type 7: Negative example (Directing live project queries to Knowledge Engine)
            {
                "record_id": "dom_007",
                "type": DomainExampleType.NEGATIVE_EXAMPLE.value,
                "category": "project_separation",
                "instruction": "What is the status of Project ABC?",
                "response": "I will check the latest project records to retrieve the current status for Project ABC.",
                "provenance": prov
            },
            # Extra representative examples for category coverage
            {
                "record_id": "dom_008",
                "type": DomainExampleType.INSTRUCTION_RESPONSE.value,
                "category": "test_prep",
                "instruction": "What is the IELTS exam structure?",
                "response": "The IELTS examination consists of four sections: Listening (30 minutes), Reading (60 minutes), Writing (60 minutes), and Speaking (11-14 minutes). Both Academic and General Training formats evaluate overall English proficiency on a 1-9 band scale.",
                "provenance": prov
            },
            {
                "record_id": "dom_009",
                "type": DomainExampleType.CONVERSATION.value,
                "category": "immigration",
                "messages": [
                    {"role": "user", "content": "What is Canada Express Entry?"},
                    {"role": "assistant", "content": "Express Entry is Canada's online application management system for skilled workers seeking permanent residency. It manages three main Federal economic immigration programs: Federal Skilled Worker (FSW), Federal Skilled Trades (FST), and Canadian Experience Class (CEC)."}
                ],
                "provenance": prov
            },
            {
                "record_id": "dom_010",
                "type": DomainExampleType.INSTRUCTION_RESPONSE.value,
                "category": "work_permit",
                "instruction": "What is OPT for international students in the US?",
                "response": "Optional Practical Training (OPT) allows eligible F1 student visa holders to work in the USA for up to 12 months in a field directly related to their major area of study. STEM degree graduates may qualify for an additional 24-month extension.",
                "provenance": prov
            }
        ]

        return records

    def run_pipeline(self, seed_records: Optional[List[Dict[str, Any]]] = None, split_ratio: float = 0.8, seed: int = 42) -> Tuple[int, int, int]:
        """Runs dataset pipeline and returns tuple (train_count, val_count, eval_count)."""
        manifest = self.process_and_save(seed_records=seed_records, split_ratio=split_ratio, seed=seed)
        return manifest["train_records"], manifest["val_records"], manifest["valid_records"]

    def process_and_save(self, seed_records: Optional[List[Dict[str, Any]]] = None, split_ratio: float = 0.8, seed: int = 42) -> Dict[str, Any]:
        """
        Processes domain data: validates, checks conflicts, splits, and saves artifacts.
        """
        raw_data = seed_records or self.generate_seed_records()

        # 1. Validate & Quarantine
        valid_records, quarantined = self.validator.validate_batch(raw_data)

        # 2. Check conflicts
        conflicts = self.conflict_checker.scan_conflicts(valid_records)

        # 3. Save conflict report
        conflict_report_path = self.data_root / "metadata" / "domain_conflicts.json"
        self.conflict_checker.save_conflict_report(conflicts, conflict_report_path)

        # 4. Shuffle & Split
        rng = random.Random(seed)
        shuffled = valid_records.copy()
        rng.shuffle(shuffled)

        split_idx = int(len(shuffled) * split_ratio)
        train_recs = shuffled[:split_idx]
        val_recs = shuffled[split_idx:]

        # 5. Save jsonl files
        train_path = self.data_root / "training" / "train.jsonl"
        val_path = self.data_root / "training" / "val.jsonl"
        golden_path = self.data_root / "evaluation" / "golden.jsonl"

        train_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_jsonl(train_recs, train_path)
        self._write_jsonl(val_recs, val_path)
        self._write_jsonl(valid_records, golden_path)

        # 6. Save Manifest
        manifest = {
            "total_raw_records": len(raw_data),
            "valid_records": len(valid_records),
            "quarantined_records": len(quarantined),
            "conflicts_detected": len(conflicts),
            "train_records": len(train_recs),
            "val_records": len(val_recs),
            "split_ratio": split_ratio,
            "seed": seed,
            "train_file": str(train_path),
            "val_file": str(val_path),
            "golden_file": str(golden_path),
        }
        manifest_path = self.data_root / "metadata" / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def _write_jsonl(self, records: List[DomainRecord], path: Path) -> None:
        """Write DomainRecords to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.model_dump_json(exclude_none=True) + "\n")
