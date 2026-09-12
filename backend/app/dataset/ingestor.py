"""
Precious Edu LLM — Dataset Ingestor

Reads raw files (.txt, .json, .jsonl) from data/raw/ directories,
verifies UTF-8 encoding, generates deterministic SHA-256 record IDs,
and constructs canonical DatasetRecord instances with source provenance.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Generator, List, Optional
from app.dataset.models import DatasetRecord, MessageRecord, RecordType, SourceProvenance

logger = logging.getLogger(__name__)


class DatasetIngestor:
    """
    Ingests raw dataset files from data/raw/ subdirectories.
    """

    def discover_raw_files(self, raw_dir: Path) -> List[Path]:
        """Discover all supported data files (.txt, .json, .jsonl) in raw_dir."""
        files: List[Path] = []
        if not raw_dir.exists():
            return files

        for ext in ["*.txt", "*.json", "*.jsonl"]:
            files.extend(raw_dir.rglob(ext))

        return sorted(files)

    def ingest_file(self, file_path: Path, source_id: Optional[str] = None) -> Generator[DatasetRecord, None, None]:
        """
        Ingest a single raw file and yield canonical DatasetRecord objects.
        """
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return

        sid = source_id or file_path.stem
        provenance = SourceProvenance(
            source_id=sid,
            source_name=file_path.name,
            source_type=file_path.parent.name,
            license="unknown",
            origin=str(file_path.resolve()),
            language="en"
        )

        ext = file_path.suffix.lower()

        if ext == ".txt":
            yield from self._ingest_txt(file_path, provenance)
        elif ext == ".jsonl":
            yield from self._ingest_jsonl(file_path, provenance)
        elif ext == ".json":
            yield from self._ingest_json(file_path, provenance)

    def _ingest_txt(self, file_path: Path, provenance: SourceProvenance) -> Generator[DatasetRecord, None, None]:
        """Ingest .txt file line by line."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f):
                    text = line.strip()
                    if not text:
                        continue
                    rec_id = self._generate_record_id(provenance.source_id, f"line_{line_idx}", text)
                    yield DatasetRecord(
                        record_id=rec_id,
                        source_id=provenance.source_id,
                        type=RecordType.PLAIN_TEXT,
                        language=provenance.language,
                        text=text,
                        provenance=provenance
                    )
        except Exception as err:
            logger.error(f"Error reading TXT file {file_path}: {err}", exc_info=True)

    def _ingest_jsonl(self, file_path: Path, provenance: SourceProvenance) -> Generator[DatasetRecord, None, None]:
        """Ingest .jsonl file line by line."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                for line_idx, line in enumerate(f):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        data = json.loads(line_str)
                        record = self._parse_json_dict(data, provenance, line_idx)
                        if record:
                            yield record
                    except Exception as parse_err:
                        logger.warning(f"Failed to parse JSONL line {line_idx} in {file_path.name}: {parse_err}")
                        # Yield malformed record for quarantine validator
                        rec_id = self._generate_record_id(provenance.source_id, f"malformed_{line_idx}", line_str)
                        yield DatasetRecord(
                            record_id=rec_id,
                            source_id=provenance.source_id,
                            type=RecordType.PLAIN_TEXT,
                            language=provenance.language,
                            text=line_str,
                            metadata={"raw_parse_error": str(parse_err)},
                            provenance=provenance
                        )
        except Exception as err:
            logger.error(f"Error reading JSONL file {file_path}: {err}", exc_info=True)

    def _ingest_json(self, file_path: Path, provenance: SourceProvenance) -> Generator[DatasetRecord, None, None]:
        """Ingest single .json file (array of records or single record object)."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = json.load(f)
                if isinstance(content, list):
                    for idx, item in enumerate(content):
                        if isinstance(item, dict):
                            record = self._parse_json_dict(item, provenance, idx)
                            if record:
                                yield record
                elif isinstance(content, dict):
                    record = self._parse_json_dict(content, provenance, 0)
                    if record:
                        yield record
        except Exception as err:
            logger.error(f"Error reading JSON file {file_path}: {err}", exc_info=True)

    def _parse_json_dict(self, data: dict, provenance: SourceProvenance, idx: int) -> Optional[DatasetRecord]:
        """Parse raw dict into DatasetRecord."""
        if "messages" in data and isinstance(data["messages"], list):
            messages: List[MessageRecord] = []
            msg_str_parts = []
            for msg_item in data["messages"]:
                if isinstance(msg_item, dict):
                    role = str(msg_item.get("role", "")).strip()
                    content = str(msg_item.get("content", "")).strip()
                    messages.append(MessageRecord(role=role, content=content))
                    msg_str_parts.append(f"{role}:{content}")

            content_repr = "||".join(msg_str_parts)
            rec_id = self._generate_record_id(provenance.source_id, f"conv_{idx}", content_repr)
            return DatasetRecord(
                record_id=rec_id,
                source_id=provenance.source_id,
                type=RecordType.CONVERSATION,
                language=provenance.language,
                messages=messages,
                metadata=data.get("metadata", {}),
                provenance=provenance
            )

        elif "text" in data and isinstance(data["text"], str):
            text = data["text"].strip()
            rec_id = self._generate_record_id(provenance.source_id, f"text_{idx}", text)
            return DatasetRecord(
                record_id=rec_id,
                source_id=provenance.source_id,
                type=RecordType.PLAIN_TEXT,
                language=provenance.language,
                text=text,
                metadata=data.get("metadata", {}),
                provenance=provenance
            )

        return None

    def _generate_record_id(self, source_id: str, item_key: str, content: str) -> str:
        """Generate deterministic SHA-256 record ID based on source, item key, and content."""
        seed_str = f"{source_id}::{item_key}::{content}"
        return hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:16]
