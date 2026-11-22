from skill_agent.skills.base import Skill, SkillCard


class DummySkill(Skill):
    def __init__(self, name: str, description: str, examples: list[str] | None = None) -> None:
        self.card = SkillCard(name=name, description=description, examples=examples or [])

    def run(self, request, llm):
        return f"{self.card.name}:{request}"


class FakeStructuredLLM:
    """Mimics `llm.with_structured_output(...)`: returns queued results, raises queued errors."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
