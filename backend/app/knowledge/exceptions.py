"""
Precious Edu LLM — Project Knowledge Engine Exceptions

Custom exception hierarchy for Excel ingestion, validation, repository,
search, and knowledge routing operations.
"""

class KnowledgeError(Exception):
    """Base exception for all Knowledge Engine errors."""
    pass


class ExcelIngestionError(KnowledgeError):
    """Raised when an Excel file cannot be read or parsed."""
    pass


class InvalidWorkbookError(ExcelIngestionError):
    """Raised when a workbook structure is corrupt or unsupported."""
    pass


class ValidationError(KnowledgeError):
    """Raised when records fail schema or field validation."""
    def __init__(self, message: str, errors: list = None):
        super().__init__(message)
        self.errors = errors or []


class KnowledgeRepositoryError(KnowledgeError):
    """Raised when database operations fail."""
    pass


class KnowledgeSearchError(KnowledgeError):
    """Raised when knowledge search query fails."""
    pass


class KnowledgeRouterError(KnowledgeError):
    """Raised when prompt intent routing fails."""
    pass
