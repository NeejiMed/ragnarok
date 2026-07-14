from pathlib import Path

from bs4 import BeautifulSoup


def extract_text(file_path: str) -> list[dict]:
    """
    Extracts plain text from TXT and Markdown files
    """
    content = Path(file_path).read_text(encoding="utf-8", errors="replace")
    return [{"content": content.strip(), "page": None, "extraction_method": "native"}]


def extract_html(file_path: str) -> list[dict]:
    """Extracts clean text from HTML files by stripping tags"""
    raw = Path(file_path).read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    # Remove script and style elements, they contain no useful content
    for tag in soup(["script", "style"]):
        tag.decompose()  # Remove the tag from the tree

    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines produced by the removal of tags
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    return [{"content": clean_text, "page": None, "extraction_method": "native"}]
