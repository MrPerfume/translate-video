from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class AppSettings:
    openai_api_key: str | None
    deepseek_api_key: str | None
    elevenlabs_api_key: str | None
    openai_base_url: str
    deepseek_base_url: str


@lru_cache(maxsize=1)
def load_settings() -> AppSettings:
    """Load API settings from .env without exposing secrets to logs.
    Result is cached for the lifetime of the process; call reload_settings()
    if the .env file changes at runtime.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()
    return AppSettings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY") or None,
        openai_base_url=(os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
        deepseek_base_url=(os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
    )


def reload_settings() -> AppSettings:
    """Clear the settings cache and reload from .env."""
    load_settings.cache_clear()
    return load_settings()

