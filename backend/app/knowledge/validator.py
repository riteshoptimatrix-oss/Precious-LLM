"""
Precious Edu LLM — Project Record Validation Module

Validates normalized canonical project records.
Checks required fields, duplicate project IDs, date logic, malformed rows,
and generates validation error reports.
"""

import logging
from typing import Any, Dict, List, Tuple

from app.knowledge.models import ProjectRecord, SourceMetadata

logger = logging.getLogger(__name__)


class RecordValidator:
    """
    Validates canonical project records prior to database ingestion.
    """

    def __init__(self, allow_missing_id: bool = False):
        self.allow_missing_id = allow_missing_id

    def validate_records(self, raw_parsed_records: List[Dict[str, Any]], mapper: Any, normalizer: Any) -> Tuple[List[ProjectRecord], List[Dict[str, Any]]]:
        """
        Processes raw parsed records through mapper, normalizer, and validator.
        Returns:
            Tuple of (valid_project_records, invalid_row_reports)
        """
        valid_records: List[ProjectRecord] = []
        invalid_reports: List[Dict[str, Any]] = []

        seen_project_ids = set()

        for idx, item in enumerate(raw_parsed_records):
            raw_row = item.get("raw_data", {})
            src_info = item.get("source", {})

            # 1. Map columns
            canonical_data, attributes = mapper.map_row(raw_row)

            # 2. Normalize data
            norm_canonical, norm_attributes = normalizer.normalize_record(canonical_data, attributes)

            # 3. Validate
            row_errors = []
            proj_id = norm_canonical.get("project_id", "").strip()
            proj_name = norm_canonical.get("project_name", "").strip()

            # Check required fields
            if not proj_id and not proj_name:
                row_errors.append("Missing both project_id and project_name")

            if not proj_id and not self.allow_missing_id:
                # Fallback project_id from name if allowed, or error
                if proj_name:
                    proj_id = f"PROJ-{idx+1:03d}"
                    norm_canonical["project_id"] = proj_id
                else:
                    row_errors.append("Missing required field: project_id")

            # Duplicate ID check in batch
            if proj_id:
                if proj_id in seen_project_ids:
                    row_errors.append(f"Duplicate project_id within dataset batch: '{proj_id}'")
                else:
                    seen_project_ids.add(proj_id)

            # Date logic check
            start_d = norm_canonical.get("start_date")
            end_d = norm_canonical.get("end_date")
            if start_d and end_d and start_d > end_d:
                row_errors.append(f"start_date ({start_d}) is after end_date ({end_d})")

            # If errors found, add to invalid reports
            if row_errors:
                invalid_reports.append({
                    "source": src_info,
                    "raw_data": raw_row,
                    "errors": row_errors
                })
                logger.warning(f"Validation failed for row {src_info.get('row')} in sheet '{src_info.get('sheet')}': {row_errors}")
                continue

            # Build canonical ProjectRecord
            source_meta = SourceMetadata(
                file=src_info.get("file", "unknown.xlsx"),
                sheet=src_info.get("sheet", "Sheet1"),
                row=src_info.get("row", 0),
                file_hash=src_info.get("file_hash", ""),
                dataset_version=src_info.get("dataset_version", "projects-v1"),
            )

            record = ProjectRecord(
                project_id=proj_id,
                project_name=proj_name or proj_id,
                client=norm_canonical.get("client") or "Unknown",
                status=norm_canonical.get("status") or "Active",
                manager=norm_canonical.get("manager") or "Unassigned",
                start_date=start_d,
                end_date=end_d,
                description=norm_canonical.get("description") or "",
                attributes=norm_attributes,
                source=source_meta,
                dataset_version=source_meta.dataset_version,
                is_active=True,
            )

            valid_records.append(record)

        return valid_records, invalid_reports
