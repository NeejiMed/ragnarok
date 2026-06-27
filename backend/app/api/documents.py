from fastapi import APIRouter, HTTPException, UploadFile

from backend.app.ingestion.schemas import IngestedDocument
from backend.app.ingestion.service import FileTooLargeError, UnsupportedFileTypeError, save_upload

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=IngestedDocument)
async def upload_document(file: UploadFile) -> IngestedDocument:
    try:
        return await save_upload(file)
    except UnsupportedFileTypeError as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    except FileTooLargeError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
