"""
Precious Edu LLM — Response Generator Compatibility Adapter

Re-exports ResponseGenerator and TemporaryResponseGenerator from app.llm
to maintain backward compatibility with any Phase 1 imports.
"""

from app.llm.base import ResponseGenerator
from app.llm.temporary_generator import TemporaryResponseGenerator

# Compatibility alias for Phase 1 code
BaseResponseGenerator = ResponseGenerator

__all__ = ["BaseResponseGenerator", "ResponseGenerator", "TemporaryResponseGenerator"]
