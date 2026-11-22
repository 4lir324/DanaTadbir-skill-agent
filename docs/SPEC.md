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
START -> shortlist -> route --+--> run_step <-+   (loops once per planned step)
                              |       |       |
                              |       +-------+--> respond --> END
                              +--> clarify  --> END
                              +--> fallback --> END
```

Nodes (see `src/skill_agent/graph.py`, diagram in `docs/graph.mmd`):
- `shortlist`: registry cards -> at most k candidates.
- `route`: LLM -> `RoutingDecision` (Pydantic), sanitized (unknown skills dropped,
  2-skill cap, first step never "chained").
- `run_step`: runs one planned skill with the user's original prompt, the step
  instruction and (if chained) the previous output.
- `respond`: chained -> last output; independent -> both, labelled; adds a note
  when skills were skipped by the limit.
- `clarify` / `fallback`: see 4.3.

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
- Provider presets (see section 7, item 6): `groq` (recommended, `qwen/qwen3.8-27b`, 15/15 on every
  run after the structured-output fix) and `hetzner` (`Qwen3.8-27B`, free while experimental;
  10 requests/min; timed out on 4 of 15 evaluation cases in the last run). Client throttles and
  retries on 429/timeouts.
- Google AI Studio (Gemma) and OpenRouter free models were tried and found unusable (README section 6).
- Constraint from the brief: free tier, <= 35B parameters.

## 5. Out of scope
Memory across requests, persistence, auth, deployment, web API,
more than 4 skills, more than 2 simultaneous skills.

## 6. Pipeline (branches, in order)

| # | Branch | Deliverable |
|---|---|---|
| 0 | `docs/spec` | `CLAUDE.md`, this spec |
| 1 | `feat/scaffold-llm` | `pyproject.toml`, config, LLM client (throttle/retry), connection check |
| 2 | `feat/registry-router` | Skill registry, shortlist, `RoutingDecision`, router |
| 3 | `feat/skills` | The four skills, safe calculator, prompt notes |
| 4 | `feat/graph-cli` | LangGraph wiring, fallback, multi-skill execution, CLI |
| 5 | `feat/evaluation` | 15-prompt eval set + runner + report |
| 5b | `fix/over-limit-notice` | Tell the user when a requested skill was skipped |
| 6 | `docs/readme` | README, spec sync |
| 7 | `feat/provider-alternatives` | Configurable structured-output method; Groq tested as an alternative provider |

Pushes are done by the repo owner (or with their explicit approval each time).

## 7. Decisions taken during implementation
1. Hetzner supports structured output and tool calling (verified). Qwen3.x
   "thinking" must be off: with it on, the routing JSON contained repeated items.
2. Shortlist retriever: lexical only (IDF-weighted stem overlap, Persian/English
   stopwords, identifier tokens not truncated). Embeddings left as an extension.
3. The router does not always keep details such as the target language in a
   step's `input`, so every skill also receives the user's original prompt.
4. Requests for more than 2 skills run the first 2 and say what was skipped.
5. Structured output: `json_mode` with the schema shown in the prompt is the default.
   `json_schema` timed out on Hetzner (after a 15/15 run) and is wrapped in an extra key on
   Groq; `function_calling` also works. The method is configurable (`LLM_STRUCTURED_OUTPUT`).
6. Provider presets (`config.py`): `hetzner` and `groq`, each reading only its own key variable.
   The provider is detected from the key present, or set by `LLM_PROVIDER`. A bare `LLM_API_KEY`
   is rejected (a Groq key once could have been sent to Hetzner's URL), and `LLM_BASE_URL`
   selects a "custom" provider that assumes nothing (no Qwen-specific parameters).

