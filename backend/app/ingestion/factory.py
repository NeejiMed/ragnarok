from enum import Enum

from backend.app.ingestion.extractors.docx_extractor import extract_docx
from backend.app.ingestion.extractors.pdf_extractor import extract_pdf


class UnsupportedFileTypeError(Exception):
    """Custom exception for unsupported file types."""

    pass


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    MARKDOWN = "md"
    HTML = "html"
    TXT = "txt"
    CSV = "csv"


class DocumentLoaderFactory:
    @staticmethod
    def get_document_type(file_name: str) -> DocumentType:
        extension = file_name.rsplit(".", 1)[-1].lower()
        try:
            return DocumentType(extension)
        except ValueError:
            raise UnsupportedFileTypeError(f"Unsupported file type: .{extension}") from None

    @staticmethod
    def load(file_path: str, document_type: DocumentType) -> list[dict]:
        if document_type == DocumentType.PDF:
            return extract_pdf(file_path)
        if document_type == DocumentType.DOCX:
            return extract_docx(file_path)
        raise NotImplementedError(f"Loader for {document_type} not yet implemented")
