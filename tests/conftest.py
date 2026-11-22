from langchain_core.messages import AIMessage

from skill_agent.skills.base import Skill, SkillCard, SkillRequest


class DummySkill(Skill):
    def __init__(self, name: str, description: str, examples: list[str] | None = None) -> None:
        self.card = SkillCard(name=name, description=description, examples=examples or [])

    def run(self, request: SkillRequest, llm) -> str:
        return f"{self.card.name}:{request.instruction}"


class FakeStructuredLLM:
    """Mimics `llm.with_structured_output(...)`: returns queued results, raises queued errors."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def with_structured_output(self, _schema, **_kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeChatLLM:
    """Plain chat fake: always answers `content`; with tools, answers `tool_calls`."""

    def __init__(self, content: str = "", tool_calls: list[dict] | None = None):
        self._message = AIMessage(content=content, tool_calls=tool_calls or [])
        self.calls = []

    def bind_tools(self, _tools, **_kwargs):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self._message
