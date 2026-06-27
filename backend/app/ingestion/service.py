import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import UploadFile

from backend.app.core.config import settings
from backend.app.ingestion.factory import DocumentLoaderFactory, UnsupportedFileTypeError
from backend.app.ingestion.schemas import DocumentStatus, IngestedDocument


class FileTooLargeError(Exception):
    """Custom exception for files that exceed the maximum allowed size."""

    pass


async def save_upload(file: UploadFile) -> IngestedDocument:
    """Save an uploaded file to the designated upload directory and return its metadata."""
    # 1. Validate extension early to avoid unnecessary file operations
    if not file.filename:
        raise ValueError("Uploaded file must have a filename.")
    document_type = DocumentLoaderFactory.get_document_type(file.filename)

    # 2. Read into memory to check size (acceptable for our size limit; for very
    #    large files in real prod you'd stream-check size via headers/chunks instead)
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise FileTooLargeError(
            f"File size {size_mb:.2f} MB exceeds the maximum allowed size of \
                  {settings.max_upload_size_mb} MB."
        )

    # 3. Unsupported file type check (redundant but explicit)
    if document_type not in DocumentLoaderFactory.get_document_type(file.filename):
        raise UnsupportedFileTypeError(f"Unsupported file type: {document_type}")

    # 3. Generate ID and save file to disk
    document_id = str(uuid.uuid4())
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_path = upload_dir / f"{document_id}_{file.filename}"
    saved_path.write_bytes(contents)

    # 4. Return our standardized schema, content stays empty until Phase 2 parses it
    return IngestedDocument(
        document_id=document_id,
        file_name=file.filename,
        document_type=document_type,
        content="",  # Placeholder; actual content extraction happens in Phase 2
        metadata={
            "saved_path": str(saved_path),
            "upload_timestamp": datetime.now(UTC).isoformat(),
            "size_mb": round(size_mb, 2),
        },
        status=DocumentStatus.PENDING,
        created_at=datetime.now(UTC),
    )
