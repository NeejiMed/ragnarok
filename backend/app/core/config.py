from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 50
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    ocr_min_text_threshold: int = 20  # chars; below this, treat page as "no text"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 32

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
