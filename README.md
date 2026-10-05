# Maintenance Triage Copilot

An evidence-gated maintenance-support prototype on public flight-recorder data: it scores
how far a flight deviates from post-maintenance engine behaviour, retrieves similar past
maintenance cases, and drafts an inspection note in which every claim cites its evidence.
When the evidence is weak it answers "inconclusive". A technician approves or rejects.

It is decision support, not an airworthiness tool, and on this dataset it is **not** a
failure predictor: see "What the data showed". Goal, architecture and scope are in
`docs/BRIEF.md`; every decision and result is logged in `docs/DECISIONS.md`.

## What the data showed (val split, test split untouched)

| Question | Result |
|---|---|
| Can flights before maintenance be told from flights after? | Yes, AUC 0.77 (accuracy 71%), but through oil pressure, which steps up by about 1.2 psi at maintenance |
| Is there signal outside the oil channels? | Hardly: AUC 0.54 on per-flight summaries, 0.56–0.57 with InceptionTime on 1 Hz data |
| Can the issue type be predicted? | Weakly (macro AUC 0.67), and most of it is still there after the repair |
| Does the unsupervised deviation score rank flights? | No, AUC 0.52 |
| Does similar-case retrieval beat "always the most frequent issue"? | No, top-1 50% against 54% |
| What does the agent do with this? | It stays silent on about 96% of flights; every note it does produce passes the citation check |

The public data has no aircraft id, no dates and no maintenance record content, so
per-aircraft baselines, trends and a split by aircraft are not possible. Those three
things exist in an operator's own data, which is where the method can be tested properly.

## Setup
```bash
cp .env.example .env        # fill in keys (none are needed for the offline demo)
uv sync --group dev --extra agent --extra api --extra app   # add --extra dl for PyTorch
uv run pytest
uv run ruff check .
```

## Data
```bash
uv run python scripts/download_ngafid.py   # 5.4 GB into data/raw/ngafid, md5 verified
```
NGAFID maintenance dataset, Yang et al. 2022, Zenodo 6624956, CC BY 4.0. A 12-flight
extract for tests is in `data/sample/`.

## Pipeline (run in this order)
```bash
uv run python scripts/make_splits.py            # folds and splits
uv run python scripts/build_features.py --all   # per-flight features (long, reads raw data)
uv run python scripts/gate1_signal.py           # F1: is there signal?
uv run python scripts/f2_tabular_baselines.py   # F2: with and without oil features
uv run python scripts/f2_oil_hypothesis.py      # F2: oil pressure around maintenance
uv run python scripts/f2_issue_class.py         # F2: issue class, with after-repair control
uv run python scripts/f3_health_score.py        # F3: normal-behaviour residuals and score
uv run python scripts/f4_case_retrieval.py      # F4: case table and retrieval evaluation
uv run python scripts/f5_agent_eval.py          # F5: agent reliability (add --llm for Gemini)
```
The sequence baseline (F2.3) runs on a GPU: `notebooks/kaggle_f2_sequence.ipynb`.
Results are written to `results/<phase>/`.

## Demo
```bash
uv run uvicorn mtc.api.app:app --port 8000      # API
uv run streamlit run app/streamlit_app.py       # UI on http://localhost:8501
```
Without `GOOGLE_API_KEY` and `PINECONE_API_KEY` the demo uses a deterministic note writer
and an in-memory case index; with them it uses Gemini and Pinecone.

## Layout
`src/mtc/data` loading, quality, phases, features, splits · `src/mtc/models` baselines,
normal behaviour, health score, sequence model · `src/mtc/retrieval` signatures, index,
search · `src/mtc/agent` schemas, evidence, gate and checks, graph, service ·
`src/mtc/api` FastAPI app and client · `app/` Streamlit · `scripts/` everything that
produces a result · `tests/` pytest.
