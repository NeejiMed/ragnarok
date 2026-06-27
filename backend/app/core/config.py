from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    upload_dir: str = "data/uploads"
    max_upload_size_mb: int = 50

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
