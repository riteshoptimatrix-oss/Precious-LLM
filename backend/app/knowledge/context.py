"""
Precious Edu LLM — Knowledge Context Formatter

Formats SearchResult objects into clean, authoritative text blocks for LLM prompt context.
Includes source provenance and explicit zero-match / missing-field notifications
to prevent model hallucinations.
"""

import logging
from typing import List, Optional

from app.knowledge.models import ProjectRecord, SearchResult

logger = logging.getLogger(__name__)

MAX_KNOWLEDGE_CONTEXT_CHARS = 1000


class KnowledgeContextFormatter:
    """
    Formats search results into structured prompt context.
    """

    def __init__(self, max_chars: int = MAX_KNOWLEDGE_CONTEXT_CHARS):
        self.max_chars = max_chars

    def format_search_result(self, search_result: SearchResult) -> str:
        """
        Formats a SearchResult object into a controlled prompt context string.
        """
        records = search_result.records
        query = search_result.query

        # 1. Zero matches
        if not records:
            query_str = query.filters.get("project_id") or query.filters.get("project_name") or query.filters.get("query") or "requested query"
            return f"Project Knowledge:\nNo matching project was found in the dataset for '{query_str}'."

        # 2. Single match detailed format
        if len(records) == 1:
            rec = records[0]
            lines = [
                "Project Knowledge:",
                f"Project ID: {rec.project_id}",
                f"Project Name: {rec.project_name}",
                f"Client: {rec.client}",
                f"Status: {rec.status}",
                f"Manager: {rec.manager}",
            ]
            if rec.start_date:
                lines.append(f"Start Date: {rec.start_date}")
            if rec.end_date:
                lines.append(f"End Date: {rec.end_date}")
            if rec.description:
                lines.append(f"Description: {rec.description}")

            # Append extra attributes if any exist
            if rec.attributes:
                for k, v in rec.attributes.items():
                    lines.append(f"{k}: {v}")

            # Source provenance
            src = rec.source
            lines.append(f"Source: {src.file}, Sheet: {src.sheet}, Row: {src.row}")

            formatted_text = "\n".join(lines)

        # 3. Multiple matches list format
        else:
            lines = [f"Project Knowledge ({len(records)} matching projects found):"]
            for idx, rec in enumerate(records, 1):
                rec_str = (
                    f"{idx}. ID: {rec.project_id} | Name: {rec.project_name} | "
                    f"Client: {rec.client} | Status: {rec.status} | Manager: {rec.manager} | "
                    f"Source: {rec.source.file} (Row {rec.source.row})"
                )
                lines.append(rec_str)
            formatted_text = "\n".join(lines)

        # Truncate text if it exceeds maximum character limit
        if len(formatted_text) > self.max_chars:
            formatted_text = formatted_text[:self.max_chars - 20] + "\n...[truncated]"

        return formatted_text
