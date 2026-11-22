"""Skill registry: the single place where skills are known to the system."""

from collections.abc import Iterable

from skill_agent.skills.base import Skill, SkillCard


class UnknownSkillError(KeyError):
    """Raised when a skill name is not registered."""


class SkillRegistry:
    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            self.register(skill)

    def register(self, skill: Skill) -> None:
        name = skill.card.name
        if name in self._skills:
            raise ValueError(f"Skill already registered: {name}")
        self._skills[name] = skill

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError:
            raise UnknownSkillError(name) from None

    def cards(self) -> list[SkillCard]:
        return [skill.card for skill in self._skills.values()]

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
