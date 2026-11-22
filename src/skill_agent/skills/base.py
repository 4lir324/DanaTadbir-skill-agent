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


class SkillRequest(BaseModel):
    """Everything a skill receives for one step of the plan."""

    original_prompt: str = Field(description="The user's raw prompt (never rewritten).")
    instruction: str = Field(description="The router's self-contained instruction for this step.")
    previous_output: str | None = Field(
        default=None, description="Result of the previous step, when this step builds on it."
    )

    def render(self) -> str:
        """Plain-text form handed to an LLM-based skill."""
        parts = [
            f"User's original request:\n{self.original_prompt}",
            f"Task for this step:\n{self.instruction}",
        ]
        if self.previous_output is not None:
            parts.append(f"Result of the previous step (work on this):\n{self.previous_output}")
        return "\n\n".join(parts)


class Skill(ABC):
    """An executable skill. Subclasses own their full prompt/tooling."""

    card: SkillCard

    @abstractmethod
    def run(self, request: SkillRequest, llm: BaseChatModel) -> str:
        """Handle one step and return the answer text."""
