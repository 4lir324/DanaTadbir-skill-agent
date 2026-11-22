"""Connection check: `python -m skill_agent.check_llm`.

Sends two tiny requests (plain chat + structured output) to verify the
configured provider works. The API key is never printed.
"""

import sys

import openai
from langchain_core.exceptions import OutputParserException
from pydantic import BaseModel

from skill_agent.config import ConfigError, load_settings
from skill_agent.llm import build_llm
from skill_agent.router import json_instruction


class Ping(BaseModel):
    language: str
    greeting: str


def main() -> int:
    try:
        settings = load_settings()
        print(f"provider={settings.provider} url={settings.base_url} model={settings.model}")
        llm = build_llm(settings)

        print("chat:", llm.invoke("Reply with the single word: pong").content)

        # Same structured-output method the router uses, so this checks what really matters.
        structured = llm.with_structured_output(Ping, method=settings.structured_output_method)
        hint = json_instruction(Ping) if settings.structured_output_method == "json_mode" else ""
        print("structured:", structured.invoke("Greet me in Persian. Fill the schema." + hint))
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except openai.OpenAIError as error:
        print(f"LLM request failed: {error}", file=sys.stderr)
        return 1
    except OutputParserException as error:
        print(f"The model's structured output was invalid: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
