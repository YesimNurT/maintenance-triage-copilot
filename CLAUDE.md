# CLAUDE.md — Maintenance Triage Copilot

Read docs/BRIEF.md (goal, architecture, scope) and docs/DECISIONS.md (what is decided)
before planning anything.

## Stack
Python 3.11 · uv · pandas / Dask / pyarrow · scikit-learn · PyTorch (training only,
extra `dl`) · Pinecone · LangGraph + LangChain + langchain-google-genai (Gemini) ·
FastAPI · Streamlit · pytest · ruff · Docker · GitHub Actions · AWS (ECR, App Runner, S3)
· LangSmith. Settings come from `src/mtc/config.py` (`get_settings()`), never hard-coded.

## Layout
- `src/mtc/data/` loading, phases, quality, splits
- `src/mtc/models/` baseline, normal_behavior, health_score
- `src/mtc/retrieval/` signatures, index, search
- `src/mtc/agent/` state, graph, nodes, schemas, prompts
- `src/mtc/api/` FastAPI app · `app/` Streamlit
- `notebooks/` exploration only; reusable code goes to `src/`
- `evals/` metrics, scenarios, results · `tests/` pytest

## How we work
- One task per session. Propose a short plan (files, functions, tests) and wait for my
  OK before writing code. Never start the next phase without my OK.
- Small, reviewable changes. Every new function gets a pytest test; run `uv run pytest`
  and `uv run ruff check .` before saying a task is done.
- Explain design choices briefly so I can defend them in an interview.
- When we decide something (thresholds, splits, classes), append it to docs/DECISIONS.md.

## Data rules
- Never read files under `data/raw/` directly in a session (4.3 GB). Write scripts that
  I run; use `data/sample/` (small extract) for tests and examples.
- Splits are by maintenance event and preferably by aircraft; add a test that train,
  val and test do not overlap.
- Normalisation statistics come from the training split only.
- Fixed random seed from settings; save every experiment to `results/<phase>/`.

## LLM rules
- The LLM receives structured evidence only (scores, residual summaries, case ids),
  never raw time series.
- Every claim in an inspection note must cite a case id or a numeric evidence field;
  otherwise the note is rejected. Weak evidence → "inconclusive".
- No real Gemini / Pinecone calls in tests; mock them.

## Long jobs
Model training, downloads and Docker builds: give me the command; I run it locally or
on Kaggle. I will paste back only the relevant output.

## Do not
- Add services, frameworks or features outside docs/BRIEF.md unless I ask.
- Hard-code keys, paths or seeds. Commit `.env` or anything under `data/raw/`.
- Claim a result that was not produced by code in this repo.
