"""Router: decides which skill(s) a prompt needs and returns a validated plan."""

import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError

from skill_agent.shortlist import DEFAULT_K, SkillRetriever, select_candidates
from skill_agent.skills.base import SkillCard

logger = logging.getLogger(__name__)

MAX_SKILLS = 2
MAX_EXAMPLES_PER_CARD = 3
MAX_ATTEMPTS = 2


class SkillStep(BaseModel):
    skill: str = Field(description="Name of one skill from the list.")
    input: str = Field(
        description=(
            "Self-contained instruction for this skill, including any text it must work on, "
            "target language or format. For a step that uses the previous step's result, "
            "give only the instruction; the previous result is attached automatically."
        )
    )
    uses_previous_output: bool = Field(
        default=False,
        description="True only for the second step when it must work on the first step's result.",
    )


class RoutingDecision(BaseModel):
    """Router output. Empty `steps` + no question means 'no skill matched'."""

    steps: list[SkillStep] = Field(
        default_factory=list,
        description="Ordered skills to run (at most 2). Empty if none clearly applies.",
    )
    clarification_question: str | None = Field(
        default=None,
        description=(
            "Set only when the user asks for a task but it is too unclear to pick a skill "
            "or the needed text is missing. Written in the user's language."
        ),
    )


_SYSTEM_TEMPLATE = """\
You are the routing component of an assistant. Decide which skills are needed \
for the user's request. The request may be in English or Persian (Farsi).

Available skills:
{skills}

Rules:
1. Use only the skills listed above. Choose 1 skill, or 2 when the request clearly asks \
for two operations (for example "summarize this and translate it to English").
2. When two skills are needed, list them in execution order. Set uses_previous_output=true \
on the second step if it must work on the first step's result.
3. Each step's `input` must be self-contained: copy the user's own text verbatim, never \
rewrite or answer it yourself.
4. Never invent skills. If nothing clearly applies, return an empty `steps` list.
5. If the user wants a task done but it is too vague to choose a skill, or the text to work \
on is missing, return no steps and write one short clarification_question in the user's language.
6. Greetings, small talk and general questions are handled by the general chat skill, \
if it is listed.
"""


def render_cards(cards: list[SkillCard]) -> str:
    lines = []
    for card in cards:
        lines.append(f"- {card.name}: {card.description}")
        for example in card.examples[:MAX_EXAMPLES_PER_CARD]:
            lines.append(f'    e.g. "{example}"')
    return "\n".join(lines)


def build_system_prompt(cards: list[SkillCard]) -> str:
    return _SYSTEM_TEMPLATE.format(skills=render_cards(cards))


def sanitize(decision: RoutingDecision, allowed: set[str]) -> RoutingDecision:
    """Drop unknown skills and enforce the 2-skill limit."""
    steps = [step for step in decision.steps if step.skill in allowed]
    if len(steps) < len(decision.steps):
        logger.warning("Router returned unknown skill(s); dropped them")
    if len(steps) > MAX_SKILLS:
        logger.warning("Router returned %d skills; keeping the first %d", len(steps), MAX_SKILLS)
        steps = steps[:MAX_SKILLS]
    if steps:
        steps[0] = steps[0].model_copy(update={"uses_previous_output": False})
    return decision.model_copy(update={"steps": steps})


class Router:
    def __init__(
        self,
        llm: BaseChatModel,
        retriever: SkillRetriever | None = None,
        k: int = DEFAULT_K,
    ) -> None:
        self._structured_llm = llm.with_structured_output(RoutingDecision)
        self._retriever = retriever
        self._k = k

    def route(self, prompt: str, cards: list[SkillCard]) -> RoutingDecision:
        """Shortlist the cards, ask the LLM, validate. Never raises on bad LLM output."""
        candidates = select_candidates(prompt, cards, self._retriever, self._k)
        if not candidates:
            return RoutingDecision()

        messages = [
            SystemMessage(build_system_prompt(candidates)),
            HumanMessage(prompt),
        ]
        allowed = {card.name for card in candidates}

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                decision = self._structured_llm.invoke(messages)
            except (ValidationError, OutputParserException, ValueError) as error:
                logger.warning("Router attempt %d failed: %s", attempt, error)
                continue
            if isinstance(decision, RoutingDecision):
                return sanitize(decision, allowed)
        return RoutingDecision()
