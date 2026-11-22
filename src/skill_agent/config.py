"""Runtime configuration, loaded from environment variables / a `.env` file."""

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

DEFAULT_BASE_URL = "https://inference.hetzner.com/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
# Hetzner's free tier allows 10 requests / 60 s; stay slightly below it.
DEFAULT_REQUESTS_PER_MINUTE = 9


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    base_url: str
    model: str
    api_key: str
    requests_per_minute: int
    disable_thinking: bool


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    """Read settings from the environment.

    A `.env` file is searched upward from the current directory. `LLM_API_KEY`
    is preferred; `HETZNER_API_KEY` is accepted as a convenience alias.
    """
    load_dotenv(find_dotenv(usecwd=True))

    api_key = _first_env("LLM_API_KEY", "HETZNER_API_KEY")
    if not api_key:
        raise ConfigError("Missing API key: set LLM_API_KEY (or HETZNER_API_KEY) in .env")

    return Settings(
        base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        api_key=api_key,
        requests_per_minute=int(
            os.environ.get("LLM_REQUESTS_PER_MINUTE", DEFAULT_REQUESTS_PER_MINUTE)
        ),
        disable_thinking=_as_bool(os.environ.get("LLM_DISABLE_THINKING"), default=True),
    )
