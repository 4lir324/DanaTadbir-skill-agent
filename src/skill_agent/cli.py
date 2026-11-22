"""Command-line interface.

    python -m skill_agent "your prompt"      one-shot
    python -m skill_agent                    interactive (no memory between prompts)
    python -m skill_agent --plan "..."       also show how the prompt was routed
    python -m skill_agent --graph            print the graph as a Mermaid diagram
"""

import argparse
import sys
from collections.abc import Sequence

import openai

from skill_agent.agent import create_agent
from skill_agent.config import ConfigError
from skill_agent.graph import AgentGraph, AgentState

EXIT_COMMANDS = {"exit", "quit", "q"}


def describe_plan(state: AgentState) -> str:
    """One line explaining how the prompt was handled."""
    path = state["path"]
    if path == "clarify":
        return "plan: needs clarification"
    if path == "fallback":
        return "plan: no skill matched -> general_chat (fallback)"
    steps = state["decision"].steps
    return "plan: " + " -> ".join(step.skill for step in steps)


def answer_prompt(agent: AgentGraph, prompt: str, show_plan: bool) -> str:
    if not prompt.strip():
        return "Please type a request."
    state = agent.invoke(prompt)
    answer = state["answer"]
    return f"{describe_plan(state)}\n\n{answer}" if show_plan else answer


def _interactive(agent: AgentGraph, show_plan: bool) -> None:
    print("Skill agent ready. Type a request (exit to quit).")
    while True:
        try:
            prompt = input("> ")
        except EOFError:
            return
        if prompt.strip().lower() in EXIT_COMMANDS:
            return
        print(answer_prompt(agent, prompt, show_plan), end="\n\n")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="skill_agent", description=__doc__.split("\n\n")[0])
    parser.add_argument("prompt", nargs="*", help="request text; omit for interactive mode")
    parser.add_argument("--plan", action="store_true", help="show which skills were selected")
    parser.add_argument("--graph", action="store_true", help="print the graph (Mermaid) and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        agent = create_agent()
        if args.graph:
            print(agent.compiled.get_graph().draw_mermaid())
        elif args.prompt:
            print(answer_prompt(agent, " ".join(args.prompt), args.plan))
        else:
            _interactive(agent, args.plan)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except openai.OpenAIError as error:
        print(f"LLM request failed: {error}", file=sys.stderr)
        return 1
    return 0
