from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
STATIC_DIR = APP_DIR / "static"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


class Settings(BaseSettings):
    ai_provider: str = "ollama"
    mock_mode: bool = False
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5vl:7b"
    ollama_timeout: int = 180
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    backend_cors_origins: str = "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174"
    database_path: str = "backend/data/app.db"
    upload_dir: str = "backend/uploads"

    model_config = SettingsConfigDict(
        env_file=(str(PROJECT_ROOT / ".env"),),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.backend_cors_origins.split(",") if item.strip()]

    @property
    def database_file(self) -> Path:
        return resolve_project_path(self.database_path)

    @property
    def upload_path(self) -> Path:
        return resolve_project_path(self.upload_dir)

    @property
    def legacy_database_file(self) -> Path | None:
        path = Path(self.database_path)
        if path.is_absolute():
            return None
        legacy = (BACKEND_DIR / path).resolve()
        return legacy if legacy != self.database_file else None

    @property
    def legacy_upload_path(self) -> Path | None:
        path = Path(self.upload_dir)
        if path.is_absolute():
            return None
        legacy = (BACKEND_DIR / path).resolve()
        return legacy if legacy != self.upload_path else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
