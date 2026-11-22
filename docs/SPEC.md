# Design Spec — Multi-Skill Agent

Status: draft v1. This file is the source of truth given to the AI assistant
(see `CLAUDE.md`) and is kept in the repo as part of the AI-usage disclosure.

## 1. Goal

A LangGraph agent that takes one free-form user prompt (English or Persian),
detects which of the supported skills it needs (at most 2), runs them, and
returns one response. Prototype quality, but clean structure.

Source brief: `docs/task/AIEngineerTask.pdf` + interviewer's voice note
(`docs/task/prj.docx`).

## 2. Skills (exactly four)

| Skill | Purpose | Notes |
|---|---|---|
| `summarizer` | Summarize the user's text | Output in the language of the input unless asked otherwise |
| `translator` | Translate the user's text | Handles fa<->en and other pairs; detects target language from the prompt |
| `calculator` | Math | **Real tool**: LLM only extracts the expression; a safe AST evaluator computes the result |
| `general_chat` | Greetings, small talk, general questions | Also the fallback |

## 3. Graph

```
 user prompt
     |
 [shortlist]   registry -> top-k candidate skills (only their short cards go to the router)
     |
 [router]      LLM -> RoutingDecision (Pydantic): ordered skills (1-2) + per-skill input + confidence
     |
 [validate]    schema OK? known skills? <=2? confident? else -> fallback
     |
 [execute]     run skill 1 -> (optionally) feed its output to skill 2
     |
 [respond]     assemble final answer
```

LangGraph nodes: `shortlist`, `router`, `execute_skill` (loops over the plan),
`fallback`, `respond`. Conditional edges after `router`.

## 4. Key design decisions

### 4.1 Scalability to N skills (the "context explosion" requirement)
The interviewer asked that the design stays sound at 10, 20, N skills.
- Skills live in a **registry**; each skill declares a small *card*
  (`name`, one-line `description`, a few `examples`) and owns its own full prompt.
- The router never sees full skill prompts, only cards.
- A **shortlist stage** reduces N cards to top-k (k=4 by default) before the
  router runs. With N <= k the stage is a pass-through, so today's 4 skills
  behave like a plain router.
- The retriever is behind an interface (`SkillRetriever`) so it can be swapped.
  Default implementation: lexical scoring over card text (no downloads, works
  offline). An embedding retriever is an optional extension (needs a model
  download, so it requires explicit user approval).
- Router prompt size is therefore O(k), not O(N). A test registers ~50 dummy
  skills and asserts the router prompt stays bounded.
- Adding a skill = one new file + one registry entry; no change to the graph.

### 4.2 Two skills in one prompt
Router emits an **ordered plan** of 1-2 steps, each with its own sub-input.
Execution is **sequential with chaining**: step 2 receives step 1's output
when the plan marks it dependent (e.g. "summarize then translate"), otherwise
both run on their own sub-input and results are concatenated.
Reason: the common multi-skill prompts are pipelines (summarize -> translate);
sequential is the simplest model that handles both cases and is easy to debug.

### 4.3 Fallback
Router output is validated. Cases:
- no skill matches / low confidence -> `general_chat` answers normally;
- ambiguous prompt (e.g. "do it" with no clear task) -> a short clarifying question;
- router JSON invalid after one retry -> `general_chat`;
- more than 2 skills returned -> keep the first 2 and log a warning.

Assumption to document in README: the voice note said "error if no skill",
the PDF said "General Chat or clarifying response". We choose a graceful
response over a hard error and say so.

### 4.4 Calculator
LLM extracts a plain math expression (Persian/Arabic-Indic digits normalized
first). Evaluation uses a whitelist-based `ast` evaluator (numbers, + - * / **
// %, parentheses, a small set of `math` functions). No `eval`. Errors
(division by zero, bad syntax) are reported to the user, never guessed.

### 4.5 Persian
- Router prompt and skill prompts are written in English, with Persian
  examples in the cards/few-shots.
- Output language rules are explicit in each skill prompt.
- Digit normalization for the calculator; RTL text handled as plain UTF-8.

### 4.6 LLM provider
OpenAI-compatible client, configured by environment variables:
`LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`.
- Primary: Hetzner Experiments Inference, `Qwen3.8-27B` (dense, 27B, free
  while experimental). Limits: 10 requests/min -> client has throttling +
  retry on 429.
- Documented alternative: OpenRouter free model `google/gemma-4-31b-it:free` (30.7B).
- Constraint from the brief: free tier, <= 35B parameters.

## 5. Out of scope
Memory across requests, persistence, auth, deployment, web API,
more than 4 skills, more than 2 simultaneous skills.

## 6. Pipeline (branches, in order)

| # | Branch | Deliverable |
|---|---|---|
| 0 | `docs/spec` | `CLAUDE.md`, this spec |
| 1 | `feat/scaffold-llm` | `pyproject.toml`, config, LLM client (throttle/retry), connection check |
| 2 | `feat/registry-router` | Skill registry, shortlist, `RoutingDecision`, router node |
| 3 | `feat/skills` | The four skills + safe calculator + unit tests |
| 4 | `feat/graph-cli` | LangGraph wiring, fallback, multi-skill execution, CLI |
| 5 | `feat/evaluation` | 10-15 prompt eval set + runner + accuracy report |
| 6 | `docs/readme` | Final README with all required sections |

Each branch is merged into `main` through a pull request. Pushes are done by
the repo owner (or with their explicit approval each time).

## 7. Open questions
1. Does the Hetzner endpoint support structured output / tool calling?
   (checked in step 1; fallback: JSON mode + Pydantic validation)
2. Shortlist retriever: lexical only, or add embeddings?
3. Prompt-improvement example for README: record v1 vs final during step 3.
