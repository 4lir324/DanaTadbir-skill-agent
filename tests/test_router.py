from conftest import DummySkill, FakeStructuredLLM
from langchain_core.exceptions import OutputParserException

from skill_agent.router import (
    MAX_SKILLS,
    Router,
    RoutingDecision,
    SkillStep,
    build_system_prompt,
    sanitize,
)
from skill_agent.shortlist import select_candidates


def _cards(n_extra=0):
    base = [
        DummySkill("summarizer", "Summarize text", ["Summarize this", "خلاصه کن"]).card,
        DummySkill("translator", "Translate text", ["Translate to French"]).card,
    ]
    extra = [
        DummySkill(f"skill_{i}", f"Does task number {i} for users", ["example a", "example b"]).card
        for i in range(n_extra)
    ]
    return base + extra


def _step(skill, **kw):
    return SkillStep(skill=skill, input="x", **kw)


def test_route_returns_validated_decision():
    decision = RoutingDecision(steps=[_step("summarizer")])
    router = Router(FakeStructuredLLM([decision]))
    assert router.route("Summarize: hello", _cards()).steps[0].skill == "summarizer"


def test_unknown_skills_dropped_and_limit_enforced():
    decision = RoutingDecision(
        steps=[_step("ghost"), _step("summarizer"), _step("translator"), _step("summarizer")]
    )
    result = sanitize(decision, {"summarizer", "translator"})
    assert [s.skill for s in result.steps] == ["summarizer", "translator"]
    assert len(result.steps) <= MAX_SKILLS


def test_first_step_never_uses_previous_output():
    decision = RoutingDecision(steps=[_step("summarizer", uses_previous_output=True)])
    assert sanitize(decision, {"summarizer"}).steps[0].uses_previous_output is False


def test_retries_once_then_succeeds():
    ok = RoutingDecision(steps=[_step("translator")])
    llm = FakeStructuredLLM([OutputParserException("bad json"), ok])
    result = Router(llm).route("Translate", _cards())
    assert result.steps[0].skill == "translator" and len(llm.calls) == 2


def test_gives_up_after_two_failures_with_empty_decision():
    llm = FakeStructuredLLM([OutputParserException("bad"), OutputParserException("bad")])
    result = Router(llm).route("Translate", _cards())
    assert result.steps == [] and result.clarification_question is None


def test_router_prompt_stays_bounded_with_many_skills():
    """The context-explosion requirement: prompt size depends on k, not on N."""
    small = build_system_prompt(_cards()[:2])
    llm = FakeStructuredLLM([RoutingDecision()])
    cards = _cards(n_extra=50)
    Router(llm).route("task number 7 please", select_candidates("task number 7 please", cards, k=4))
    sent_prompt = llm.calls[0][0].content
    assert sent_prompt.count("\n- ") <= 4  # only k cards rendered
    assert len(sent_prompt) < 3 * len(small) + 500


def test_no_candidates_means_no_llm_call():
    llm = FakeStructuredLLM([])
    result = Router(llm).route("zzzz qqqq", [])
    assert result.steps == [] and llm.calls == []
