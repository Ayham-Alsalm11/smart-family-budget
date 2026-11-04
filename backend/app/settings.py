"""Configuration from environment variables and backend/.env."""
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = BACKEND_DIR / "data" / "fba.db"

Weekday = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    groq_api_key: SecretStr | None = None
    jwt_secret: SecretStr | None = None
    database_url: str = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    model_main: str = "openai/gpt-oss-120b"
    model_small: str = "openai/gpt-oss-20b"
    agent_today: date | None = None
    week_start: Weekday = "saturday"
    trend_threshold: float = 0.3
    corrections_enabled: bool = False
    seed_manager_password: SecretStr | None = None
    seed_member_password: SecretStr | None = None

    @property
    def groq_key_present(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key.get_secret_value())

    def today(self) -> date:
        """AGENT_TODAY when set (reproducible evaluation, §7.4), otherwise the real date."""
        return self.agent_today or date.today()


@lru_cache
def get_settings() -> Settings:
    return Settings()
