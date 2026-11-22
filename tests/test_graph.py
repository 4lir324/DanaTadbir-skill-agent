import pytest
from conftest import DummySkill

from skill_agent.graph import SKILL_FAILED_MESSAGE, AgentGraph
from skill_agent.registry import SkillRegistry
from skill_agent.router import RoutingDecision, SkillStep
from skill_agent.skills.base import Skill, SkillCard, SkillRequest


class RecordingSkill(Skill):
    """Echoes what it received so tests can check data flow between steps."""

    def __init__(self, name, description="does things", examples=None):
        self.card = SkillCard(name=name, description=description, examples=examples or [])
        self.requests: list[SkillRequest] = []

    def run(self, request, llm):
        self.requests.append(request)
        return f"{self.card.name}({request.previous_output or request.instruction})"


class BrokenSkill(RecordingSkill):
    def run(self, request, llm):
        raise RuntimeError("boom")


class StubRouter:
    def __init__(self, decision):
        self.decision = decision
        self.seen_candidates = None

    def route(self, prompt, candidates):
        self.seen_candidates = candidates
        return self.decision


def _agent(decision, *skills, k=4):
    registry = SkillRegistry(skills or [RecordingSkill("summarizer"), RecordingSkill("general_chat")])
    router = StubRouter(decision)
    return AgentGraph(registry, router, llm=None, k=k), router, registry


def _step(skill, text="x", chained=False):
    return SkillStep(skill=skill, input=text, uses_previous_output=chained)


def test_single_skill_path():
    agent, _, registry = _agent(RoutingDecision(steps=[_step("summarizer", "do A")]))
    state = agent.invoke("original")
    assert state["path"] == "skills" and state["answer"] == "summarizer(do A)"
    request = registry.get("summarizer").requests[0]
    assert request.original_prompt == "original"  # skills always get the untouched prompt


def test_two_chained_skills_pass_output_forward_and_return_last():
    decision = RoutingDecision(
        steps=[_step("summarizer", "sum"), _step("translator", "to English", chained=True)]
    )
    agent, _, registry = _agent(decision, RecordingSkill("summarizer"), RecordingSkill("translator"))
    state = agent.invoke("p")
    assert registry.get("translator").requests[0].previous_output == "summarizer(sum)"
    assert state["answer"] == "translator(summarizer(sum))"


def test_two_independent_skills_are_both_returned_with_labels():
    decision = RoutingDecision(steps=[_step("calculator", "2+2"), _step("translator", "hi")])
    agent, _, registry = _agent(decision, RecordingSkill("calculator"), RecordingSkill("translator"))
    state = agent.invoke("p")
    assert registry.get("translator").requests[0].previous_output is None
    assert state["answer"] == "[calculator]\ncalculator(2+2)\n\n[translator]\ntranslator(hi)"


def test_note_is_added_when_extra_skills_were_skipped():
    decision = RoutingDecision(steps=[_step("calculator", "2+2")])
    decision.mark_skipped(["translator"])
    agent, _, _ = _agent(decision, RecordingSkill("calculator"))
    answer = agent.invoke("p")["answer"]
    assert answer.startswith("calculator(2+2)") and "not done: translator" in answer


def test_no_note_when_nothing_was_skipped():
    agent, _, _ = _agent(RoutingDecision(steps=[_step("summarizer", "x")]))
    assert agent.invoke("p")["answer"] == "summarizer(x)"


def test_no_match_falls_back_to_general_chat():
    agent, _, registry = _agent(RoutingDecision())
    state = agent.invoke("how is the moon?")
    assert state["path"] == "fallback"
    assert registry.get("general_chat").requests[0].original_prompt == "how is the moon?"


def test_clarification_question_is_returned_without_running_skills():
    agent, _, registry = _agent(RoutingDecision(clarification_question="Which text?"))
    state = agent.invoke("translate this")
    assert state["path"] == "clarify" and state["answer"] == "Which text?"
    assert all(not skill.requests for skill in (registry.get("summarizer"), registry.get("general_chat")))


def test_failing_skill_degrades_gracefully():
    agent, _, _ = _agent(RoutingDecision(steps=[_step("calculator")]), BrokenSkill("calculator"))
    state = agent.invoke("p")
    assert state["answer"] == SKILL_FAILED_MESSAGE.format(skill="calculator")


def test_router_only_sees_the_shortlist_when_many_skills_exist():
    skills = [RecordingSkill("general_chat", "chat")] + [
        RecordingSkill(f"s{i}", f"unique{i} topic{i}", [f"keyword{i}"]) for i in range(30)
    ]
    agent, router, _ = _agent(RoutingDecision(), *skills, k=3)
    agent.invoke("please use keyword7")
    assert [c.name for c in router.seen_candidates] == ["s7"]
    assert len(router.seen_candidates) <= 3


def test_graph_can_be_rendered_as_mermaid():
    agent, _, _ = _agent(RoutingDecision())
    mermaid = agent.compiled.get_graph().draw_mermaid()
    for node in ("shortlist", "route", "run_step", "respond", "clarify", "fallback"):
        assert node in mermaid


def test_dummy_skill_still_satisfies_the_contract():
    assert DummySkill("a", "b").run(SkillRequest(original_prompt="p", instruction="i"), None) == "a:i"


@pytest.mark.parametrize("prompt", ["", "   "])
def test_cli_rejects_empty_prompt_without_calling_the_graph(prompt):
    from skill_agent.cli import answer_prompt

    assert answer_prompt(agent=None, prompt=prompt, show_plan=False) == "Please type a request."
