import pandas as pd


def extract_csv(file_path: str) -> list[dict]:
    """
    Extracts data from a CSV file as individual Documents.
    Each row becomes: "ColumnName: value" for each column in the row.
    Handles encoding fallback ( UTF-8 -> Latin-1).
    """
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    results = []
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        row_text = ", ".join(
            f"{col}: {val}" for col, val in row.items() if pd.notna(val) and str(val).strip()
        )
        if row_text.strip():  # Only add non-empty rows
            results.append({"content": row_text, "page": row_number, "extraction_method": "native"})

    return results
