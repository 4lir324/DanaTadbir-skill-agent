"""Pure scoring helpers for the evaluation (no LLM, unit-tested).

An *outcome* describes what the agent did with a prompt:
    "summarizer"              one skill
    "summarizer>translator"   two skills, in this order (chained)
    "calculator+translator"   two independent skills (order does not matter)
    "fallback"                no skill matched, general chat answered
    "clarify"                 the agent asked a clarification question
"""

from collections.abc import Iterable

from skill_agent.graph import AgentState


def outcome_of(state: AgentState) -> str:
    """Describe how the agent handled a prompt, in the notation above."""
    if state["path"] != "skills":
        return state["path"]
    steps = state["decision"].steps
    names = [step.skill for step in steps]
    if len(steps) == 2:
        separator = ">" if steps[1].uses_previous_output else "+"
        return separator.join(names)
    return names[0]


def matches(outcome: str, expected: str) -> bool:
    """True if `outcome` equals `expected`; '+' outcomes compare without order."""
    if "+" in expected or "+" in outcome:
        return set(outcome.split("+")) == set(expected.split("+")) and "+" in outcome
    return outcome == expected


def is_correct(outcome: str, acceptable: Iterable[str]) -> bool:
    return any(matches(outcome, option) for option in acceptable)


def answer_ok(answer: str, must_contain: str | None) -> bool | None:
    """None when the case has no answer check, otherwise whether the answer contains the text."""
    return None if must_contain is None else must_contain in answer
