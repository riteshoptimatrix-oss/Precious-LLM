"""
Precious Edu LLM — Domain Record Validator & Security Scanner

Validates domain training examples across all 7 types, checks required fields,
scans for security credentials/secrets, detects duplicates, and quarantines invalid records.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Tuple

from app.domain.schema import DomainExampleType, DomainRecord, QuarantinedRecord

logger = logging.getLogger(__name__)


class DomainValidator:
    """
    Validates domain records prior to training dataset inclusion.
    """

    SECRET_PATTERNS = [
        re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),  # API keys like OpenAI
        re.compile(r"AIza[0-9A-Za-z-_]{35}", re.IGNORECASE), # Google API keys
        re.compile(r"mongodb(?:\+srv)?://[^:]+:[^@]+@", re.IGNORECASE), # DB URIs with passwords
        re.compile(r"bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE), # Bearer tokens
        re.compile(r"password\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE), # Hardcoded passwords
    ]

    def __init__(self, max_length_chars: int = 4000):
        self.max_length_chars = max_length_chars

    def scan_for_secrets(self, text: str) -> List[str]:
        """Scans text string for security credentials or secrets."""
        found_secrets = []
        for pattern in self.SECRET_PATTERNS:
            if pattern.search(text):
                found_secrets.append(f"Detected potential secret matching pattern: {pattern.pattern[:30]}...")
        return found_secrets

    def validate_record(self, raw_data: Dict[str, Any]) -> Tuple[Optional[DomainRecord], Optional[QuarantinedRecord]]:
        """
        Validates a single raw dictionary against DomainRecord schema and safety rules.
        """
        rec_id = str(raw_data.get("record_id", f"rec_{hashlib.md5(str(raw_data).encode()).hexdigest()[:8]}"))
        if "record_id" not in raw_data:
            raw_data["record_id"] = rec_id
        errors = []

        try:
            record = DomainRecord(**raw_data)
        except Exception as e:
            return None, QuarantinedRecord(
                record_id=rec_id,
                reason="Schema parsing error",
                errors=[str(e)],
                raw_data=raw_data
            )

        # 1. Type specific field validation
        rec_type = record.type
        if rec_type in (DomainExampleType.INSTRUCTION_RESPONSE, DomainExampleType.CLARIFICATION, DomainExampleType.AMBIGUITY, DomainExampleType.NEGATIVE_EXAMPLE):
            if not record.instruction or not record.instruction.strip():
                errors.append("Missing required field: instruction")
            if not record.response or not record.response.strip():
                errors.append("Missing required field: response")

        elif rec_type == DomainExampleType.TERMINOLOGY:
            if not record.term or not record.term.strip():
                errors.append("Missing required field: term")
            if not record.definition or not record.definition.strip():
                errors.append("Missing required field: definition")

        elif rec_type == DomainExampleType.SERVICE:
            if not record.service_name or not record.service_name.strip():
                errors.append("Missing required field: service_name")
            if not record.response or not record.response.strip():
                errors.append("Missing required field: response")

        elif rec_type == DomainExampleType.CONVERSATION:
            if not record.messages or not isinstance(record.messages, list) or len(record.messages) < 2:
                errors.append("Conversation must contain at least 2 messages")
            else:
                for idx, m in enumerate(record.messages):
                    r = m.get("role")
                    c = m.get("content")
                    if r not in ("user", "assistant", "system"):
                        errors.append(f"Message index {idx} has invalid role: '{r}'")
                    if not c or not c.strip():
                        errors.append(f"Message index {idx} has empty content")

        # 2. Length check
        full_text = f"{record.instruction or ''} {record.response or ''} {record.definition or ''}"
        if record.messages:
            full_text += " ".join(m.get("content", "") for m in record.messages)

        if len(full_text) > self.max_length_chars:
            errors.append(f"Exceeds maximum character length limit ({len(full_text)} > {self.max_length_chars})")

        # 3. Security Scan
        secrets_found = self.scan_for_secrets(full_text)
        if secrets_found:
            errors.extend(secrets_found)

        if errors:
            return None, QuarantinedRecord(
                record_id=rec_id,
                reason="Validation failed",
                errors=errors,
                raw_data=raw_data
            )

        return record, None

    def validate_and_deduplicate(self, raw_records: List[Dict[str, Any]]) -> Tuple[List[DomainRecord], List[QuarantinedRecord]]:
        return self.validate_batch(raw_records)

    def validate_batch(self, raw_records: List[Dict[str, Any]]) -> Tuple[List[DomainRecord], List[QuarantinedRecord]]:
        """
        Validates a list of raw records, removing duplicates and quarantining invalid items.
        """
        valid_records: List[DomainRecord] = []
        quarantined: List[QuarantinedRecord] = []
        seen_hashes = set()

        for item in raw_records:
            rec, q_report = self.validate_record(item)
            if q_report:
                quarantined.append(q_report)
                continue

            # Deduplication hash check
            content_str = f"{rec.type}:{rec.instruction}:{rec.response}:{rec.term}:{rec.service_name}"
            chash = hashlib.sha256(content_str.encode()).hexdigest()

            if chash in seen_hashes:
                quarantined.append(QuarantinedRecord(
                    record_id=rec.record_id,
                    reason="Duplicate record content",
                    errors=["Duplicate content hash within batch"],
                    raw_data=item
                ))
            else:
                seen_hashes.add(chash)
                valid_records.append(rec)

        return valid_records, quarantined
