"""LLM client factory.

Any OpenAI-compatible endpoint works (Hetzner, OpenRouter, ...). The client is
rate limited client-side and retries on HTTP 429.
"""

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI

from skill_agent.config import Settings, load_settings

MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 120


def build_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Create the chat model used by the router and the skills."""
    settings = settings or load_settings()

    # Qwen3.x "thinks" by default, which is slow and made structured routing
    # output unreliable in our tests (see docs/SPEC.md, section 4.6).
    extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}} if settings.disable_thinking else None
    )

    return ChatOpenAI(
        base_url=settings.base_url,
        model=settings.model,
        api_key=settings.api_key,
        temperature=0,
        max_retries=MAX_RETRIES,
        timeout=REQUEST_TIMEOUT_SECONDS,
        rate_limiter=InMemoryRateLimiter(requests_per_second=settings.requests_per_minute / 60),
        extra_body=extra_body,
    )
