"""Skill contract: a small public *card* plus a private implementation."""

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field


class SkillCard(BaseModel):
    """What the router (and shortlist) is allowed to know about a skill.

    Keep it short: the router prompt grows with the number of cards shown, so
    the full skill prompt must never live here.
    """

    name: str
    description: str = Field(description="One line: what the skill does and when to use it.")
    examples: list[str] = Field(
        default_factory=list, description="A few example user requests (English and Persian)."
    )


class Skill(ABC):
    """An executable skill. Subclasses own their full prompt/tooling."""

    card: SkillCard

    @abstractmethod
    def run(self, request: str, llm: BaseChatModel) -> str:
        """Handle `request` (a self-contained instruction, possibly with text) and return the answer."""
