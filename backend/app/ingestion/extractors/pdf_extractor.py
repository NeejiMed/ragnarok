import fitz  # PyMuPDF
import pytesseract
from PIL import Image

from backend.app.core.config import settings

pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def extract_pdf(file_path: str) -> list[dict]:
    """
    Extract text from PDF, page by page.
    Returns a list of dicts: {"content": str, "page": int, "extraction_method": str}
    """
    results = []
    pdf: fitz.Document = fitz.open(file_path)

    try:
        for page_number, page in enumerate(pdf, start=1):  # type: ignore
            text = page.get_text().strip()

            if len(text) >= settings.ocr_min_text_threshold:
                results.append(
                    {"content": text, "page": page_number, "extraction_method": "native"}
                )

            else:
                # Fallback: render page to image, run OCR
                ocr_text = _ocr_page(page)
                results.append(
                    {"content": ocr_text, "page": page_number, "extraction_method": "ocr"}
                )
    finally:
        pdf.close()

    return results


def _ocr_page(page: "fitz.Page", zoom: float = 2.0) -> str:
    """Renders a PDF page to an image and runs the Tesseract OCR on it"""
    matrix = fitz.Matrix(zoom, zoom)  # zoom=2.0 roughly doubles resolution: better OCR accuracy)
    pixmap = page.get_pixmap(matrix=matrix)
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    return pytesseract.image_to_string(image).strip()
