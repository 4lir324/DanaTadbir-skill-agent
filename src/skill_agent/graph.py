"""LangGraph workflow.

    START -> shortlist -> route --+--> run_step <-+  (loops once per planned step)
                                  |       |       |
                                  |       +-------+--> respond --> END
                                  +--> clarify  --> END
                                  +--> fallback --> END
"""

import logging
from typing import Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from skill_agent.catalog import FALLBACK_SKILL
from skill_agent.registry import SkillRegistry
from skill_agent.router import Router, RoutingDecision
from skill_agent.shortlist import DEFAULT_K, SkillRetriever, select_candidates
from skill_agent.skills.base import SkillCard, SkillRequest

logger = logging.getLogger(__name__)

SKILL_FAILED_MESSAGE = "Sorry, the {skill} skill failed to handle this request."
# Bilingual on purpose: the router does not report the user's language.
SKIPPED_NOTE = (
    "(Note: at most 2 skills run per request; not done: {skills}.)\n"
    "(توجه: در هر درخواست حداکثر ۲ مهارت اجرا می‌شود؛ انجام نشد: {skills}.)"
)


class AgentState(TypedDict, total=False):
    prompt: str
    candidates: list[SkillCard]
    decision: RoutingDecision
    step_index: int
    outputs: list[str]
    answer: str
    path: Literal["skills", "fallback", "clarify"]


class AgentGraph:
    """Builds and owns the compiled graph. Nodes are methods, so wiring reads top to bottom."""

    def __init__(
        self,
        registry: SkillRegistry,
        router: Router,
        llm: BaseChatModel,
        retriever: SkillRetriever | None = None,
        k: int = DEFAULT_K,
        fallback_skill: str = FALLBACK_SKILL,
    ) -> None:
        self._registry = registry
        self._router = router
        self._llm = llm
        self._retriever = retriever
        self._k = k
        self._fallback_skill = fallback_skill
        self.compiled = self._build()

    def invoke(self, prompt: str) -> AgentState:
        return self.compiled.invoke({"prompt": prompt, "step_index": 0, "outputs": []})

    # ---- wiring -----------------------------------------------------------------------

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("shortlist", self._shortlist)
        graph.add_node("route", self._route)
        graph.add_node("run_step", self._run_step)
        graph.add_node("respond", self._respond)
        graph.add_node("clarify", self._clarify)
        graph.add_node("fallback", self._fallback)

        graph.add_edge(START, "shortlist")
        graph.add_edge("shortlist", "route")
        graph.add_conditional_edges("route", self._after_route, ["run_step", "clarify", "fallback"])
        graph.add_conditional_edges("run_step", self._after_step, ["run_step", "respond"])
        for terminal in ("respond", "clarify", "fallback"):
            graph.add_edge(terminal, END)
        return graph.compile()

    # ---- nodes ------------------------------------------------------------------------

    def _shortlist(self, state: AgentState) -> AgentState:
        cards = select_candidates(state["prompt"], self._registry.cards(), self._retriever, self._k)
        return {"candidates": cards}

    def _route(self, state: AgentState) -> AgentState:
        return {"decision": self._router.route(state["prompt"], state["candidates"])}

    def _run_step(self, state: AgentState) -> AgentState:
        index = state["step_index"]
        step = state["decision"].steps[index]
        outputs = state["outputs"]
        request = SkillRequest(
            original_prompt=state["prompt"],
            instruction=step.input,
            previous_output=outputs[-1] if step.uses_previous_output and outputs else None,
        )
        return {"outputs": [*outputs, self._run_skill(step.skill, request)], "step_index": index + 1}

    def _respond(self, state: AgentState) -> AgentState:
        steps, outputs = state["decision"].steps, state["outputs"]
        chained = len(steps) == 2 and steps[1].uses_previous_output
        if len(outputs) == 1 or chained:
            answer = outputs[-1]  # a chained step already contains the earlier result
        else:
            answer = "\n\n".join(f"[{s.skill}]\n{out}" for s, out in zip(steps, outputs, strict=True))
        if state["decision"].skipped:
            answer += "\n\n" + SKIPPED_NOTE.format(skills=", ".join(state["decision"].skipped))
        return {"answer": answer, "path": "skills"}

    def _clarify(self, state: AgentState) -> AgentState:
        return {"answer": state["decision"].clarification_question, "path": "clarify"}

    def _fallback(self, state: AgentState) -> AgentState:
        request = SkillRequest(original_prompt=state["prompt"], instruction=state["prompt"])
        return {"answer": self._run_skill(self._fallback_skill, request), "path": "fallback"}

    # ---- edges ------------------------------------------------------------------------

    @staticmethod
    def _after_route(state: AgentState) -> Literal["run_step", "clarify", "fallback"]:
        decision = state["decision"]
        if decision.steps:
            return "run_step"
        return "clarify" if decision.clarification_question else "fallback"

    @staticmethod
    def _after_step(state: AgentState) -> Literal["run_step", "respond"]:
        more = state["step_index"] < len(state["decision"].steps)
        return "run_step" if more else "respond"

    # ---- helpers ----------------------------------------------------------------------

    def _run_skill(self, name: str, request: SkillRequest) -> str:
        """Run one skill; a failing skill degrades to a message instead of crashing the agent."""
        try:
            return self._registry.get(name).run(request, self._llm)
        except Exception:
            logger.exception("Skill %s failed", name)
            return SKILL_FAILED_MESSAGE.format(skill=name)
