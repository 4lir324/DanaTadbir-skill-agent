# CLAUDE.md

Instructions for the AI coding assistant (Claude Code) working on this repo.
Kept in the repository as part of the AI-usage disclosure required by the task.

## Project
Multi-skill LangGraph agent (summarizer, translator, calculator, general chat)
with Persian support. Full design: `docs/SPEC.md`. Task brief: `docs/task/`.

## Rules
- Follow `docs/SPEC.md`. If a decision changes, update the spec in the same PR.
- Python 3.12, type hints everywhere, Pydantic for structured data.
- Clean code: small functions, one responsibility per module, no dead code.
- Exactly 4 skills, at most 2 per prompt. Do not add memory, DB, auth or a web API.
- Calculator must use the safe AST evaluator, never `eval` and never LLM guessing.
- Never print, log or commit secrets. Secrets live in `.env` (gitignored).
- LLM must be free-tier and <= 35B parameters.

## Git
- Work on feature branches named in the SPEC pipeline; small, logical commits
  using conventional prefixes (`feat:`, `fix:`, `docs:`, `test:`, `chore:`).
- Never push, force-push, or change the active GitHub account without the
  owner's explicit approval at that moment. Say when a push is needed.
- Never download anything (models, datasets, packages beyond declared deps)
  without asking first.

## Commands (filled in as the project grows)
- Install: `pip install -e .[dev]`
- Run: `python -m skill_agent "<prompt>"`
- Test: `pytest`
- Eval: `python -m evals.run`
