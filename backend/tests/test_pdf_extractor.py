from pathlib import Path

import pytest

from backend.app.ingestion.extractors.pdf_extractor import extract_pdf
from backend.app.ingestion.factory import (
    DocumentLoaderFactory,
    DocumentType,
    UnsupportedFileTypeError,
)

TEST_FILES_DIR = Path(__file__).parent / "fixtures"


def make_text_pdf(path: str, pages: list[str]) -> None:
    """Generates a synthetic PDF with one page per string in `pages`."""
    import fitz

    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_normal_pdf_extraction():
    pdf_path = TEST_FILES_DIR / "basic-text.pdf"

    extracted_content = extract_pdf(str(pdf_path))

    assert len(extracted_content) == 1
    assert extracted_content[0]["extraction_method"] == "native"
    assert extracted_content[0]["page"] == 1
    assert "Sample" in extracted_content[0]["content"]


def test_multiple_pages_pdf_extraction():
    pdf_path = TEST_FILES_DIR / "multi-page.pdf"

    extracted_content = extract_pdf(str(pdf_path))

    assert len(extracted_content) == 5
    for i, item in enumerate(extracted_content, start=1):
        assert item["page"] == i
        assert item["extraction_method"] == "native"


def test_invalid_path_pdf_extraction():
    fake_path = "backend/tests/does_not_exist.pdf"

    with pytest.raises(RuntimeError):
        extract_pdf(fake_path)


def test_unsupported_file_type():
    with pytest.raises(UnsupportedFileTypeError):
        DocumentLoaderFactory.get_document_type("unsupported_file.xyz")


def test_get_document_type_pdf_via_path_string():
    assert DocumentLoaderFactory.get_document_type("basic-text.pdf") == DocumentType.PDF


def test_ocr_pdf_extraction():
    pdf_path = TEST_FILES_DIR / "image.pdf"

    extracted_content = extract_pdf(str(pdf_path))

    assert len(extracted_content) >= 1
    assert extracted_content[0]["extraction_method"] == "ocr"
    assert extracted_content[0]["page"] == 1
