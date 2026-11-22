import pytest

from skill_agent import config
from skill_agent.config import ConfigError, load_settings
from skill_agent.llm import build_llm

ENV_NAMES = [
    "LLM_API_KEY",
    "HETZNER_API_KEY",
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


def test_missing_key_raises():
    with pytest.raises(ConfigError):
        load_settings()


def test_defaults_and_hetzner_alias(monkeypatch):
    monkeypatch.setenv("HETZNER_API_KEY", "k")
    settings = load_settings()
    assert settings.api_key == "k"
    assert settings.base_url == config.DEFAULT_BASE_URL
    assert settings.model == config.DEFAULT_MODEL
    assert settings.disable_thinking is True


def test_llm_api_key_wins_over_alias(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "primary")
    monkeypatch.setenv("HETZNER_API_KEY", "alias")
    assert load_settings().api_key == "primary"


def test_overrides(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "other")
    monkeypatch.setenv("LLM_DISABLE_THINKING", "false")
    settings = load_settings()
    assert settings.model == "other"
    assert settings.disable_thinking is False


def test_build_llm_sets_thinking_off_and_retries(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    llm = build_llm()
    assert llm.model_name == config.DEFAULT_MODEL
    assert llm.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert llm.max_retries == 5


def test_structured_output_method_default_and_validation(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    assert load_settings().structured_output_method == "json_mode"
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT", "function_calling")
    assert load_settings().structured_output_method == "function_calling"
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT", "nonsense")
    with pytest.raises(ConfigError):
        load_settings()
