"""Domain exceptions for Knowledge Copilot."""


class CopilotDomainError(Exception):
    """Base domain exception."""


class DocumentNotFoundError(CopilotDomainError):
    """Raised when a document is not found."""


class DocumentValidationError(CopilotDomainError):
    """Raised when a document fails validation."""


class OversizedDocumentError(DocumentValidationError):
    """Raised when a document exceeds the configured size limit."""


class UnsupportedFormatError(DocumentValidationError):
    """Raised when a document format/extension is unsupported."""


class ExtractionError(CopilotDomainError):
    """Raised when text extraction fails (e.g., scanned PDF without text layer)."""


class IndexingError(CopilotDomainError):
    """Raised when vector or metadata indexing fails."""


class SessionNotFoundError(CopilotDomainError):
    """Raised when a conversation session is not found."""


class SessionConcurrencyError(CopilotDomainError):
    """Raised when concurrent session modifications collide."""


class ToolExecutionError(CopilotDomainError):
    """Raised when a tool execution fails."""


class AdversarialInputError(CopilotDomainError):
    """Raised when malicious or unsafe input is detected."""
