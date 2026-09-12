"""
Precious Edu LLM — Safe Data Normalization Module

Normalizes safe field types (whitespace, Unicode NFKC, date formatting,
numeric representation) without rewriting business identifiers or values.
"""

import datetime
import logging
import re
import unicodedata
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DataNormalizer:
    """
    Normalizes raw project values safely.
    """

    @staticmethod
    def normalize_string(val: Any) -> str:
        """Trims whitespace and applies NFKC Unicode normalization."""
        if val is None:
            return ""
        s = str(val).strip()
        return unicodedata.normalize("NFKC", s)

    @staticmethod
    def normalize_date(val: Any) -> Optional[str]:
        """
        Converts date objects or date strings into ISO format YYYY-MM-DD.
        Returns None if date cannot be parsed safely.
        """
        if val is None or val == "":
            return None

        if isinstance(val, (datetime.date, datetime.datetime)):
            return val.strftime("%Y-%m-%d")

        s = str(val).strip()
        if not s:
            return None

        # Standard YYYY-MM-DD
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return s

        # Try common date formats
        formats = [
            "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S", "%b %d, %Y", "%d %b %Y"
        ]
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(s, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        logger.warning(f"Could not normalize date value: {val}")
        return s  # Fallback to string representation if unrecognized

    @staticmethod
    def normalize_record(canonical_data: Dict[str, Any], attributes: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Normalizes canonical record fields and extra attributes.
        """
        norm_canonical = {}

        # String fields
        for field in ["project_id", "project_name", "client", "status", "manager", "description"]:
            val = canonical_data.get(field)
            norm_canonical[field] = DataNormalizer.normalize_string(val)

        # Dates
        norm_canonical["start_date"] = DataNormalizer.normalize_date(canonical_data.get("start_date"))
        norm_canonical["end_date"] = DataNormalizer.normalize_date(canonical_data.get("end_date"))

        # Normalize attributes dict values
        norm_attributes = {}
        for k, v in attributes.items():
            clean_k = DataNormalizer.normalize_string(k)
            if isinstance(v, (datetime.date, datetime.datetime)):
                norm_attributes[clean_k] = DataNormalizer.normalize_date(v)
            elif isinstance(v, str):
                norm_attributes[clean_k] = DataNormalizer.normalize_string(v)
            else:
                norm_attributes[clean_k] = v

        return norm_canonical, norm_attributes
