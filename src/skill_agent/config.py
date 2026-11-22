"""Runtime configuration, loaded from environment variables / a `.env` file.

A *provider preset* bundles the endpoint, model and quirks of a known provider, and each
preset reads only its own API key, so a key can never be sent to the wrong provider.
The provider is chosen by `LLM_PROVIDER`, otherwise detected from which key is present.
Any individual setting can still be overridden with its own variable.
"""

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

# How the router asks the LLM for structured output. Providers differ in what they support
# well: Groq wraps `json_schema` replies in an extra key, and Hetzner's `json_schema` path
# started timing out. `json_mode` worked on both; Pydantic validation + retry guard the output.
STRUCTURED_OUTPUT_METHODS = ("json_schema", "function_calling", "json_mode")
DEFAULT_STRUCTURED_OUTPUT_METHOD = "json_mode"
DEFAULT_REQUESTS_PER_MINUTE = 9


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or inconsistent."""


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    key_variable: str | None  # None: the generic LLM_API_KEY is the only key variable
    base_url: str | None
    model: str | None
    requests_per_minute: int
    # chat_template_kwargs is vLLM/Qwen specific: Groq and Google reject it with HTTP 400.
    disable_thinking: bool


PROVIDERS: dict[str, ProviderPreset] = {
    # Free while experimental; allows 10 requests / 60 s, stay slightly below it.
    "hetzner": ProviderPreset(
        "hetzner", "HETZNER_API_KEY", "https://inference.hetzner.com/v1", "Qwen/Qwen3.8-27B", 9, True
    ),
    # Free tier: 30 requests/min but only 8K tokens/min, which is the real limit.
    "groq": ProviderPreset(
        "groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "qwen/qwen3.8-27b", 10, False
    ),
}
# Used when LLM_BASE_URL is set without LLM_PROVIDER: nothing is assumed about the endpoint.
CUSTOM = ProviderPreset("custom", None, None, None, DEFAULT_REQUESTS_PER_MINUTE, False)


@dataclass(frozen=True)
class Settings:
    provider: str
    base_url: str
    model: str
    api_key: str
    requests_per_minute: int
    disable_thinking: bool
    structured_output_method: str


def _env(name: str) -> str | None:
    return os.environ.get(name) or None


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _choose_preset() -> ProviderPreset:
    name = _env("LLM_PROVIDER")
    if name:
        if name.lower() not in PROVIDERS:
            raise ConfigError(f"LLM_PROVIDER must be one of: {', '.join(PROVIDERS)}")
        return PROVIDERS[name.lower()]
    if _env("LLM_BASE_URL"):
        return CUSTOM
    for preset in PROVIDERS.values():
        if _env(preset.key_variable):
            return preset
    if _env("LLM_API_KEY"):
        raise ConfigError(
            "LLM_API_KEY needs LLM_PROVIDER (" + ", ".join(PROVIDERS) + ") or LLM_BASE_URL, "
            "so the key is never sent to the wrong provider"
        )
    keys = " or ".join(preset.key_variable for preset in PROVIDERS.values())
    raise ConfigError(f"No API key found: set {keys} in .env")


def _api_key(preset: ProviderPreset) -> str:
    key = _env("LLM_API_KEY") or (_env(preset.key_variable) if preset.key_variable else None)
    if not key:
        variable = preset.key_variable or "LLM_API_KEY"
        raise ConfigError(f"Missing API key for provider '{preset.name}': set {variable} in .env")
    return key


def load_settings() -> Settings:
    """Read settings from the environment (a `.env` file is searched upward from the cwd)."""
    load_dotenv(find_dotenv(usecwd=True))

    preset = _choose_preset()
    api_key = _api_key(preset)

    model = _env("LLM_MODEL") or preset.model
    if not model:
        raise ConfigError("LLM_MODEL is required when LLM_BASE_URL is set without LLM_PROVIDER")

    method = os.environ.get("LLM_STRUCTURED_OUTPUT", DEFAULT_STRUCTURED_OUTPUT_METHOD)
    if method not in STRUCTURED_OUTPUT_METHODS:
        raise ConfigError(
            f"LLM_STRUCTURED_OUTPUT must be one of {', '.join(STRUCTURED_OUTPUT_METHODS)}"
        )

    return Settings(
        provider=preset.name,
        base_url=_env("LLM_BASE_URL") or preset.base_url,
        model=model,
        api_key=api_key,
        requests_per_minute=int(
            os.environ.get("LLM_REQUESTS_PER_MINUTE", preset.requests_per_minute)
        ),
        disable_thinking=_as_bool(os.environ.get("LLM_DISABLE_THINKING"), preset.disable_thinking),
        structured_output_method=method,
    )
