from conftest import FakeChatLLM

from skill_agent.catalog import FALLBACK_SKILL, default_registry
from skill_agent.skills.base import SkillRequest
from skill_agent.skills.calculator import Calculator
from skill_agent.skills.general_chat import GeneralChat
from skill_agent.skills.summarizer import Summarizer
from skill_agent.skills.translator import Translator


def _request(**kwargs):
    base = {"original_prompt": "orig", "instruction": "do it"}
    return SkillRequest(**{**base, **kwargs})


def test_default_registry_has_exactly_four_skills():
    registry = default_registry()
    assert {card.name for card in registry.cards()} == {
        "summarizer",
        "translator",
        "calculator",
        "general_chat",
    }
    assert FALLBACK_SKILL in registry


def test_request_render_includes_original_prompt_and_previous_output():
    text = _request(original_prompt="خلاصه کن و ترجمه کن", previous_output="PREV").render()
    assert "خلاصه کن و ترجمه کن" in text and "PREV" in text and "do it" in text


def test_llm_skills_return_stripped_model_text_and_send_both_messages():
    for skill in (Summarizer(), Translator()):
        llm = FakeChatLLM(content="  result  \n")
        assert skill.run(_request(), llm) == "result"
        system, human = llm.calls[0]
        assert "Rules" in system.content and "orig" in human.content


def test_general_chat_sends_only_the_original_prompt():
    llm = FakeChatLLM(content="hi")
    assert GeneralChat().run(_request(original_prompt="سلام"), llm) == "hi"
    assert llm.calls[0][1].content == "سلام"


def _tool_call(expression):
    return {"name": "calculate", "args": {"expression": expression}, "id": "1"}


def test_calculator_uses_real_evaluator_not_llm_text():
    # The model's own text claims a wrong answer; only the tool result may count.
    llm = FakeChatLLM(content="the answer is 999", tool_calls=[_tool_call("12 * 345")])
    assert Calculator().run(_request(), llm) == "12 * 345 = 4140"


def test_calculator_handles_multiple_calls_and_errors():
    llm = FakeChatLLM(tool_calls=[_tool_call("2^10"), _tool_call("1/0")])
    answer = Calculator().run(_request(), llm)
    assert "2^10 = 1024" in answer and "error: Division by zero" in answer


def test_calculator_without_tool_call_does_not_guess():
    answer = Calculator().run(_request(), FakeChatLLM(content="It is 42"))
    assert "42" not in answer
