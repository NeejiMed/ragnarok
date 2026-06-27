import pytest

from backend.app.ingestion.factory import (
    DocumentLoaderFactory,
    DocumentType,
    UnsupportedFileTypeError,
)


def test_get_document_type_pdf():
    assert DocumentLoaderFactory.get_document_type("report.pdf") == DocumentType.PDF


def test_get_document_type_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError, match=r"Unsupported file type: \.exe"):
        DocumentLoaderFactory.get_document_type("file.exe")
