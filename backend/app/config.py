from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    backend_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    database_path: str = "backend/data/app.db"
    upload_dir: str = "backend/uploads"

    model_config = SettingsConfigDict(env_file=(".env", "../.env"), env_file_encoding="utf-8")

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.backend_cors_origins.split(",") if item.strip()]

    @property
    def database_file(self) -> Path:
        return Path(self.database_path)

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
