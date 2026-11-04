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
    jwt_expire_days: int = 7
    cors_origins: str = "http://localhost:5173"  # comma-separated
    login_max_failures: int = 5
    login_lock_minutes: int = 5
    invite_days: int = 7
    database_url: str = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"
    model_main: str = "openai/gpt-oss-120b"
    model_small: str = "openai/gpt-oss-20b"
    agent_model: Literal["main", "small"] = "main"  # the model behind POST /chat
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    llm_max_steps: int = 6
    llm_max_tokens: int = 1024
    llm_timeout: float = 30.0
    memory_messages: int = 10  # 0 switches the conversation memory off
    memory_hours: int = 24
    memory_max_tokens: int = 600
    tool_mode: Literal["native", "json"] = "native"
    agent_today: date | None = None
    week_start: Weekday = "saturday"
    trend_threshold: float = 0.3
    corrections_enabled: bool = False
    grounding_repair: bool = True  # one repair attempt when a reply has ungrounded numbers (§6.9)
    seed_manager_password: SecretStr | None = None
    seed_member_password: SecretStr | None = None

    def model_id(self, name: str) -> str:
        """'main' or 'small' → the Groq model ID (§6.3 registry)."""
        return {"main": self.model_main, "small": self.model_small}[name]

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def groq_key_present(self) -> bool:
        return bool(self.groq_api_key and self.groq_api_key.get_secret_value())

    def today(self) -> date:
        """AGENT_TODAY when set (reproducible evaluation, §7.4), otherwise the real date."""
        return self.agent_today or date.today()


@lru_cache
def get_settings() -> Settings:
    return Settings()
