import json
from pathlib import Path

import pytest

from evals.scoring import answer_ok, is_correct, matches, outcome_of
from skill_agent.router import RoutingDecision, SkillStep


def _state(*steps, path="skills"):
    return {"path": path, "decision": RoutingDecision(steps=list(steps))}


def test_outcome_notation():
    one = SkillStep(skill="summarizer", input="x")
    chained = SkillStep(skill="translator", input="y", uses_previous_output=True)
    independent = SkillStep(skill="calculator", input="z")
    assert outcome_of(_state(one)) == "summarizer"
    assert outcome_of(_state(one, chained)) == "summarizer>translator"
    assert outcome_of(_state(one, independent)) == "summarizer+calculator"
    assert outcome_of({"path": "fallback"}) == "fallback"
    assert outcome_of({"path": "clarify"}) == "clarify"


@pytest.mark.parametrize(
    ("outcome", "expected", "result"),
    [
        ("summarizer>translator", "summarizer>translator", True),
        ("translator>summarizer", "summarizer>translator", False),  # chain order matters
        ("calculator+translator", "translator+calculator", True),  # independent: any order
        ("calculator>translator", "calculator+translator", False),  # chained != independent
        ("summarizer", "summarizer", True),
        ("summarizer", "translator", False),
    ],
)
def test_matches(outcome, expected, result):
    assert matches(outcome, expected) is result


def test_is_correct_accepts_any_listed_option():
    assert is_correct("fallback", ["general_chat", "fallback"])
    assert not is_correct("calculator", ["general_chat", "fallback"])


def test_answer_ok():
    assert answer_ok("x = 746", "746") is True
    assert answer_ok("x = 7", "746") is False
    assert answer_ok("anything", None) is None


def test_dataset_is_well_formed():
    cases = json.loads((Path(__file__).parent.parent / "evals" / "dataset.json").read_text("utf-8"))
    assert 10 <= len(cases) <= 15
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(case["expected"] and case["prompt"].strip() for case in cases)
    assert {"clear_en", "clear_fa", "multi", "ambiguous"} <= {case["category"] for case in cases}
