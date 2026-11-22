from skill_agent import cli
from skill_agent.config import ConfigError
from skill_agent.router import RoutingDecision, SkillStep


class FakeAgent:
    def __init__(self, state):
        self.state = state

    def invoke(self, _prompt):
        return self.state


def test_describe_plan_variants():
    steps = [SkillStep(skill="summarizer", input="x"), SkillStep(skill="translator", input="y")]
    skills = {"path": "skills", "decision": RoutingDecision(steps=steps)}
    assert cli.describe_plan(skills) == "plan: summarizer -> translator"
    assert "fallback" in cli.describe_plan({"path": "fallback"})
    assert "clarification" in cli.describe_plan({"path": "clarify"})


def test_answer_prompt_with_and_without_plan():
    agent = FakeAgent({"path": "fallback", "answer": "hello"})
    assert cli.answer_prompt(agent, "hi", show_plan=False) == "hello"
    with_plan = cli.answer_prompt(agent, "hi", show_plan=True)
    assert with_plan.endswith("hello") and with_plan.startswith("plan:")


def test_main_one_shot_prints_answer(monkeypatch, capsys):
    monkeypatch.setattr(cli, "create_agent", lambda: FakeAgent({"path": "fallback", "answer": "ok"}))
    assert cli.main(["سلام", "دنیا"]) == 0
    assert capsys.readouterr().out.strip() == "ok"


def test_main_reports_configuration_error(monkeypatch, capsys):
    def fail():
        raise ConfigError("Missing API key")

    monkeypatch.setattr(cli, "create_agent", fail)
    assert cli.main(["hi"]) == 2
    assert "Missing API key" in capsys.readouterr().err
