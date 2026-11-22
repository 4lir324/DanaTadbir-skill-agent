import pytest

from skill_agent import config
from skill_agent.config import ConfigError, load_settings
from skill_agent.llm import build_llm

ENV_NAMES = [
    "LLM_PROVIDER",
    "LLM_API_KEY",
    "HETZNER_API_KEY",
    "GROQ_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_REQUESTS_PER_MINUTE",
    "LLM_DISABLE_THINKING",
    "LLM_STRUCTURED_OUTPUT",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    # Do not let a real .env on disk leak into the tests.
    monkeypatch.setattr(config, "load_dotenv", lambda *_a, **_k: None)
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_no_key_at_all_names_the_options():
    with pytest.raises(ConfigError, match="HETZNER_API_KEY or GROQ_API_KEY"):
        load_settings()


def test_hetzner_is_detected_from_its_key(monkeypatch):
    monkeypatch.setenv("HETZNER_API_KEY", "h")
    settings = load_settings()
    assert settings.provider == "hetzner"
    assert settings.base_url == "https://inference.hetzner.com/v1"
    assert settings.model == "Qwen/Qwen3.8-27B"
    assert settings.disable_thinking is True
    assert settings.structured_output_method == "json_mode"


def test_only_a_groq_key_just_works(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    settings = load_settings()
    assert settings.provider == "groq"
    assert settings.api_key == "g"
    assert settings.base_url == "https://api.groq.com/openai/v1"
    assert settings.model == "qwen/qwen3.8-27b"
    assert settings.disable_thinking is False  # Groq rejects chat_template_kwargs
    assert settings.requests_per_minute == 10


def test_both_keys_default_to_hetzner(monkeypatch):
    monkeypatch.setenv("HETZNER_API_KEY", "h")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    settings = load_settings()
    assert settings.provider == "hetzner" and settings.api_key == "h"


def test_explicit_provider_uses_only_its_own_key(monkeypatch):
    """A key must never travel to another provider's endpoint."""
    monkeypatch.setenv("HETZNER_API_KEY", "h")
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    settings = load_settings()
    assert settings.api_key == "g" and "groq" in settings.base_url


def test_explicit_provider_without_its_key_fails_and_names_the_variable(monkeypatch):
    monkeypatch.setenv("HETZNER_API_KEY", "h")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        load_settings()


def test_generic_key_alone_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        load_settings()


def test_generic_key_with_a_provider(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    assert load_settings().api_key == "k"


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        load_settings()


def test_custom_endpoint_needs_key_and_model_and_assumes_nothing_else(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        load_settings()
    monkeypatch.setenv("LLM_API_KEY", "k")
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        load_settings()
    monkeypatch.setenv("LLM_MODEL", "some-model")
    settings = load_settings()
    assert settings.provider == "custom" and settings.base_url == "https://example.test/v1"
    assert settings.disable_thinking is False


def test_individual_variables_override_the_preset(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    monkeypatch.setenv("LLM_MODEL", "other")
    monkeypatch.setenv("LLM_DISABLE_THINKING", "true")
    monkeypatch.setenv("LLM_REQUESTS_PER_MINUTE", "3")
    settings = load_settings()
    assert settings.model == "other"
    assert settings.disable_thinking is True
    assert settings.requests_per_minute == 3


def test_build_llm_sets_thinking_off_and_retries_for_hetzner(monkeypatch):
    monkeypatch.setenv("HETZNER_API_KEY", "h")
    llm = build_llm()
    assert llm.model_name == "Qwen/Qwen3.8-27B"
    assert llm.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert llm.max_retries == 5


def test_build_llm_sends_no_thinking_parameter_to_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "g")
    assert build_llm().extra_body is None


def test_structured_output_method_default_and_validation(monkeypatch):
    monkeypatch.setenv("HETZNER_API_KEY", "h")
    assert load_settings().structured_output_method == "json_mode"
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT", "function_calling")
    assert load_settings().structured_output_method == "function_calling"
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT", "nonsense")
    with pytest.raises(ConfigError):
        load_settings()
