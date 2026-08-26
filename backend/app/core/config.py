import sys

from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_tesseract_cmd() -> str:
    if sys.platform == "win32":
        return r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return "tesseract"


class Settings(BaseSettings):
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 50
    tesseract_cmd: str = _default_tesseract_cmd()
    ocr_min_text_threshold: int = 20
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ragnarok_documents"
    mlflow_tracking_uri: str = "http://localhost:5000"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
