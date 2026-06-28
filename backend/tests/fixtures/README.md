# Test Fixtures

- `basic-text.pdf` — single page, native embedded text. Contains the word "Sample".
- `multi-page.pdf` — 5 pages, native embedded text, used to verify page-number tracking.
- `image.pdf` — single page, image-only (no embedded text layer), used to verify OCR fallback triggers correctly.

These are committed (not generated at test-time) because reliably faking a
scanned/image-only PDF in code is fragile across environments. Native-text
PDFs are still preferable to generate programmatically when possible — see
`make_text_pdf()` in `test_pdf_extractor.py` (currently unused but kept for
reference/future tests).
