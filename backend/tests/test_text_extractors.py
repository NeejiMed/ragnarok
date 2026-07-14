from pathlib import Path

from backend.app.ingestion.extractors.csv_extractor import extract_csv
from backend.app.ingestion.extractors.text_extractor import extract_html, extract_text


def make_text_file(path: str, lines: list[str]) -> None:
    """Generates a synthetic text file with one line per string in `lines`."""
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def make_html_file(path: str, html_content: str) -> None:
    """Generates a synthetic HTML file with the provided HTML content."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)


def make_csv_file(path: str, rows: list[list[str]]) -> None:
    """Generates a synthetic CSV file with the provided rows."""
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def make_markdown_file(path: str, lines: list[str]) -> None:
    """Generates a synthetic Markdown file with one line per string in `lines`."""
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def test_plain_text_extraction():
    text_path = Path(__file__).parent / "fixtures" / "basic-text.txt"
    make_text_file(str(text_path), ["Hello, World!", "This is a test."])
    extracted_content = extract_text(str(text_path))

    assert len(extracted_content) == 1
    assert extracted_content[0]["extraction_method"] == "native"
    assert extracted_content[0]["page"] is None
    assert "Hello, World!" in extracted_content[0]["content"]
    assert "This is a test." in extracted_content[0]["content"]


def test_html_extraction():
    html_path = Path(__file__).parent / "fixtures" / "basic-html.html"
    make_html_file(
        str(html_path),
        """
        <html>
        <head><style>body { color: red; }</style></head>
        <body>
            <script>alert('xss')</script>
            <p>Hello, World!</p>
            <h1>This is a test.</h1>
        </body>
        </html>""",
    )
    extracted_content = extract_html(str(html_path))

    assert len(extracted_content) == 1
    assert extracted_content[0]["extraction_method"] == "native"
    assert extracted_content[0]["page"] is None
    assert "Hello, World!" in extracted_content[0]["content"]
    assert "This is a test." in extracted_content[0]["content"]
    # Critical: script and style content must be stripped
    assert "alert" not in extracted_content[0]["content"]
    assert "color: red" not in extracted_content[0]["content"]


def test_csv_extraction():
    csv_path = Path(__file__).parent / "fixtures" / "basic-data.csv"
    make_csv_file(str(csv_path), [["Name", "Age"], ["Alice", "30"], ["Bob", "25"]])
    extracted_content = extract_csv(str(csv_path))

    assert len(extracted_content) == 2
    assert extracted_content[0]["extraction_method"] == "native"
    assert extracted_content[0]["page"] == 1
    assert "Name: Alice, Age: 30" in extracted_content[0]["content"]
    assert "Name: Bob, Age: 25" in extracted_content[1]["content"]


def test_markdown_extraction():
    markdown_path = Path(__file__).parent / "fixtures" / "basic-markdown.md"
    make_markdown_file(str(markdown_path), ["# Hello, World!", "This is a test."])
    extracted_content = extract_text(str(markdown_path))

    assert len(extracted_content) == 1
    assert extracted_content[0]["extraction_method"] == "native"
    assert extracted_content[0]["page"] is None
    assert "# Hello, World!" in extracted_content[0]["content"]
    assert "This is a test." in extracted_content[0]["content"]
