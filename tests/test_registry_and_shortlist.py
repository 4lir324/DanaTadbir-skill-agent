import pytest
from conftest import DummySkill

from skill_agent.registry import SkillRegistry, UnknownSkillError
from skill_agent.shortlist import LexicalRetriever, select_candidates
from skill_agent.text import normalize, tokenize


def test_registry_register_get_and_duplicates():
    registry = SkillRegistry([DummySkill("a", "first")])
    assert "a" in registry and len(registry) == 1
    assert registry.get("a").card.name == "a"
    with pytest.raises(ValueError):
        registry.register(DummySkill("a", "again"))
    with pytest.raises(UnknownSkillError):
        registry.get("missing")


def test_normalize_unifies_persian_and_digits():
    assert normalize("كتاب ۱۲۳ ٤٥") == "کتاب 123 45"
    assert tokenize("می‌خوام") == ["می", "خوام"]


def _cards():
    return [
        DummySkill("summarizer", "Summarize text", ["خلاصه کن این متن رو"]).card,
        DummySkill("translator", "Translate text between languages", ["ترجمه کن"]).card,
        DummySkill("calculator", "Do math calculations", ["محاسبه کن 2+2"]).card,
        DummySkill("general_chat", "Chat and general questions", ["سلام"]).card,
    ]


def test_pass_through_when_few_skills():
    cards = _cards()
    assert select_candidates("anything", cards, k=4) == cards


def test_lexical_retriever_picks_relevant_cards_in_two_languages():
    cards = _cards() + [DummySkill(f"extra_{i}", f"unrelated thing {i}").card for i in range(10)]
    english = select_candidates("please translate this text", cards, k=3)
    persian = select_candidates("لطفا این رو ترجمه کن", cards, k=3)
    assert english[0].name == "translator"
    assert persian[0].name == "translator"


def test_no_overlap_returns_empty():
    assert LexicalRetriever().select("zzzz qqqq", _cards(), k=2) == []
