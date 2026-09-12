"""
Precious Edu LLM — Explicit Column Mapping Module

Maps raw Excel column header strings to canonical ProjectRecord fields using
deterministic rules. Extra unmapped columns are preserved under attributes.
Does NOT use LLMs or nondeterministic guessing.
"""

import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


class ColumnMapper:
    """
    Deterministic column mapper for Excel headers.
    """

    # Canonical mapping dictionary: canonical_field -> list of candidate header aliases (case-insensitive)
    DEFAULT_MAPPINGS = {
        "project_id": [
            "project id", "project_id", "projectid", "project code", "project_code",
            "id", "p_id", "pcode", "proj id", "code"
        ],
        "project_name": [
            "project name", "project_name", "projectname", "project title", "project_title",
            "project", "title", "name"
        ],
        "client": [
            "client", "client name", "client_name", "customer", "account", "organization"
        ],
        "status": [
            "status", "project status", "project_status", "state", "phase"
        ],
        "manager": [
            "manager", "project manager", "project_manager", "pm", "lead", "owner", "assignee"
        ],
        "start_date": [
            "start date", "start_date", "startdate", "commencement date", "created date"
        ],
        "end_date": [
            "end date", "end_date", "enddate", "completion date", "target date", "due date"
        ],
        "description": [
            "description", "project description", "details", "summary", "notes", "overview"
        ],
    }

    def __init__(self, custom_mappings: Dict[str, list] = None):
        self.mappings = self.DEFAULT_MAPPINGS.copy()
        if custom_mappings:
            self.mappings.update(custom_mappings)

        # Build reverse lookup: lowercase_header -> canonical_field
        self._reverse_map = {}
        for canonical_key, aliases in self.mappings.items():
            for alias in aliases:
                self._reverse_map[alias.lower().strip()] = canonical_key

    def map_row(self, raw_row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Maps a raw Excel row dictionary into:
        1. Canonical fields dict ({project_id, project_name, client, status, manager, ...})
        2. Attributes dict for leftover fields ({budget, priority, ...})
        """
        canonical_data: Dict[str, Any] = {}
        attributes: Dict[str, Any] = {}

        for raw_header, val in raw_row.items():
            cleaned_header = str(raw_header).strip()
            lower_header = cleaned_header.lower()

            canonical_key = self._reverse_map.get(lower_header)
            if canonical_key:
                # Do not overwrite if canonical key was already mapped by a higher-priority alias
                if canonical_key not in canonical_data:
                    canonical_data[canonical_key] = val
                else:
                    attributes[cleaned_header] = val
            else:
                attributes[cleaned_header] = val

        return canonical_data, attributes
