from __future__ import annotations

import os
from dataclasses import dataclass
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


def load_settings() -> AppSettings:
    """Load API settings from .env without exposing secrets to logs."""
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv()
    return AppSettings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY") or None,
        openai_base_url=(os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/"),
        deepseek_base_url=(os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/"),
    )

