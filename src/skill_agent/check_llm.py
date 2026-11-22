"""Connection check: `python -m skill_agent.check_llm`.

Sends two tiny requests (plain chat + structured output) to verify the
configured provider works. The API key is never printed.
"""

from pydantic import BaseModel

from skill_agent.config import load_settings
from skill_agent.llm import build_llm


class Ping(BaseModel):
    language: str
    greeting: str


def main() -> None:
    settings = load_settings()
    print(f"provider={settings.base_url} model={settings.model}")
    llm = build_llm(settings)

    reply = llm.invoke("Reply with the single word: pong")
    print("chat:", reply.content)

    structured = llm.with_structured_output(Ping)
    result = structured.invoke("Greet me in Persian. Fill the schema.")
    print("structured:", result)


if __name__ == "__main__":
    main()
