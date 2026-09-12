"""
Precious Edu LLM — Custom Exception Classes

Provides application-specific exception types for structured error handling.
"""


class PreciousAIException(Exception):
    """Base exception for Precious AI application errors."""
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class SessionNotFoundError(PreciousAIException):
    """Raised when a requested session is not found in MongoDB."""
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session with ID '{session_id}' not found.",
            status_code=404
        )


class DatabaseConnectionError(PreciousAIException):
    """Raised when MongoDB is unreachable or database operations fail."""
    def __init__(self, details: str = "Database connection failed."):
        super().__init__(
            message=f"Database error: {details}",
            status_code=503
        )


class InvalidMessageError(PreciousAIException):
    """Raised when message content or role validation fails."""
    def __init__(self, reason: str):
        super().__init__(
            message=f"Invalid message format: {reason}",
            status_code=400
        )
