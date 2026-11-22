"""The default skill catalog. Add a new skill here and nowhere else."""

from skill_agent.registry import SkillRegistry
from skill_agent.skills.calculator import Calculator
from skill_agent.skills.general_chat import GeneralChat
from skill_agent.skills.summarizer import Summarizer
from skill_agent.skills.translator import Translator

FALLBACK_SKILL = GeneralChat.card.name


def default_registry() -> SkillRegistry:
    """The four skills required by the task."""
    return SkillRegistry([Summarizer(), Translator(), Calculator(), GeneralChat()])
