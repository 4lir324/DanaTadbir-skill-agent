"""Composition root: wires settings, LLM, registry, router and graph together."""

from skill_agent.catalog import default_registry
from skill_agent.config import Settings, load_settings
from skill_agent.graph import AgentGraph
from skill_agent.llm import build_llm
from skill_agent.router import Router


def create_agent(settings: Settings | None = None) -> AgentGraph:
    llm = build_llm(settings or load_settings())
    return AgentGraph(registry=default_registry(), router=Router(llm), llm=llm)
