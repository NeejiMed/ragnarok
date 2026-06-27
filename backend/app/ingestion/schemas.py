from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel

from backend.app.ingestion.factory import DocumentType


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    FAILED = "failed"


class IngestedDocument(BaseModel):
    """Schema for an ingested document."""

    document_id: str  # Unique identifier for the document
    file_name: str  # Original file name of the document
    document_type: DocumentType  # Type of the document (e.g., PDF, DOCX, etc.)
    content: str  # Extracted raw text
    metadata: dict[str, Any]  # Additional metadata related to the document
    status: DocumentStatus  # Status of the document ingestion process
    created_at: datetime  # Timestamp of when the document was created
