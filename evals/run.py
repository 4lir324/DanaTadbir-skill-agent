"""Run the evaluation set against the live model: `python -m evals.run [--tag NAME]`.

Writes evals/results/latest.json and REPORT.md (or latest-NAME.json / REPORT-NAME.md with --tag,
so runs against different providers do not overwrite each other).
Makes ~2-3 LLM calls per prompt, throttled to the provider's rate limit.
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

import openai

from evals.scoring import answer_ok, is_correct, outcome_of
from skill_agent.agent import create_agent
from skill_agent.config import load_settings

HERE = Path(__file__).parent
DATASET = HERE / "dataset.json"
RESULTS_DIR = HERE / "results"
PROMPT_PREVIEW = 60


def run_case(agent, case: dict) -> dict:
    """Run one case. An LLM/network failure counts as a failed case instead of aborting the run."""
    try:
        state = agent.invoke(case["prompt"])
    except openai.OpenAIError as error:
        logging.getLogger(__name__).warning("Case %s failed: %s", case["id"], error)
        return {
            "id": case["id"],
            "category": case["category"],
            "prompt": case["prompt"],
            "expected": case["expected"],
            "outcome": f"error ({type(error).__name__})",
            "routing_correct": False,
            "answer_ok": answer_ok("", case.get("answer_contains")),
            "answer": "",
        }
    outcome = outcome_of(state)
    return {
        "id": case["id"],
        "category": case["category"],
        "prompt": case["prompt"],
        "expected": case["expected"],
        "outcome": outcome,
        "routing_correct": is_correct(outcome, case["expected"]),
        "answer_ok": answer_ok(state["answer"], case.get("answer_contains")),
        "answer": state["answer"],
    }


def summarize(results: list[dict]) -> dict:
    by_category = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result["routing_correct"])
    checked = [r["answer_ok"] for r in results if r["answer_ok"] is not None]
    return {
        "routing_accuracy": sum(r["routing_correct"] for r in results) / len(results),
        "routing_correct": sum(r["routing_correct"] for r in results),
        "total": len(results),
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in by_category.items()},
        "calculator_answers_correct": f"{sum(checked)}/{len(checked)}",
    }


def render_report(results: list[dict], summary: dict, model: str) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"- Date: {date.today()}  |  Model: `{model}`",
        f"- **Routing accuracy: {summary['routing_correct']}/{summary['total']} "
        f"({summary['routing_accuracy']:.0%})**",
        f"- Calculator answers correct: {summary['calculator_answers_correct']}",
        "",
        "| Category | Correct |",
        "|---|---|",
        *(f"| {category} | {score} |" for category, score in summary["by_category"].items()),
        "",
        "| # | Category | Prompt | Acceptable | Got | OK |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        prompt = r["prompt"].replace("\n", " ").replace("|", "/")
        if len(prompt) > PROMPT_PREVIEW:
            prompt = prompt[:PROMPT_PREVIEW] + "..."
        mark = "yes" if r["routing_correct"] else "**NO**"
        lines.append(
            f"| {r['id']} | {r['category']} | {prompt} | {' / '.join(r['expected'])} "
            f"| {r['outcome']} | {mark} |"
        )
    return "\n".join(lines) + "\n"


def _result_paths(tag: str | None) -> tuple[Path, Path]:
    suffix = f"-{tag}" if tag else ""
    return RESULTS_DIR / f"latest{suffix}.json", RESULTS_DIR / f"REPORT{suffix}.md"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="evals.run")
    parser.add_argument("--tag", help="suffix for result files, e.g. the provider name")
    args = parser.parse_args(argv)
    json_path, report_path = _result_paths(args.tag)

    settings = load_settings()
    agent = create_agent(settings)
    cases = json.loads(DATASET.read_text(encoding="utf-8"))

    results = []
    for case in cases:
        result = run_case(agent, case)
        status = "ok " if result["routing_correct"] else "XX "
        print(f"{status}#{result['id']:>2} {result['category']:<12} -> {result['outcome']}")
        results.append(result)

    summary = summarize(results)
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = {"model": settings.model, "summary": summary, "results": results}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(render_report(results, summary, settings.model), encoding="utf-8")
    print(f"\nRouting accuracy: {summary['routing_correct']}/{summary['total']}")


if __name__ == "__main__":
    main()
