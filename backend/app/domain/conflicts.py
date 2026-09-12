"""
Precious Edu LLM — Domain Conflict Detection Module

Scans domain training records for contradictory information or conflicting guidelines.
Generates `domain_conflicts.json` report prior to training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List
from app.domain.schema import DomainRecord

logger = logging.getLogger(__name__)


class DomainConflictChecker:
    """
    Scans domain records for contradictory statements.
    """

    def scan_conflicts(self, records: List[DomainRecord]) -> List[Dict[str, Any]]:
        """
        Scans a list of DomainRecords for terminology or service contradictions.
        Returns a list of conflict report dicts.
        """
        conflicts = []
        term_map: Dict[str, DomainRecord] = {}

        for rec in records:
            if rec.type == "terminology" and rec.term:
                term_key = rec.term.strip().lower()
                if term_key in term_map:
                    prev_rec = term_map[term_key]
                    if prev_rec.definition.strip().lower() != rec.definition.strip().lower():
                        conflicts.append({
                            "conflict_type": "terminology_contradiction",
                            "term": rec.term,
                            "record_a": {
                                "record_id": prev_rec.record_id,
                                "source": prev_rec.provenance.source_name,
                                "definition": prev_rec.definition
                            },
                            "record_b": {
                                "record_id": rec.record_id,
                                "source": rec.provenance.source_name,
                                "definition": rec.definition
                            },
                            "description": f"Conflicting definitions for term '{rec.term}'",
                            "resolution": "Flagged for review — manual or provenance priority required"
                        })
                else:
                    term_map[term_key] = rec

        return conflicts

    def save_conflict_report(self, conflicts: List[Dict[str, Any]], output_path: str | Path) -> None:
        """Saves conflict report to JSON file."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "total_conflicts": len(conflicts),
            "has_conflicts": len(conflicts) > 0,
            "conflicts": conflicts
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved domain conflict report ({len(conflicts)} conflicts) to {p}")
