# Maintenance Triage Copilot — Project Brief

## One line
NGAFID → temporal health modeling → fleet prioritization → interpretable
symptom/residual profile → historical case retrieval → evidence/confidence gate →
Gemini inspection note → technician approval.

## Question the system answers
"Which aircraft should we inspect today, and what should the technician check first?"

## Why
Built on public data as a prototype of a maintenance-support system an OEM or operator
could run on its own HUMS/avionics data, maintenance records, ADs and service bulletins.
The method transfers; the data does not.

## Data
NGAFID aviation maintenance dataset (Yang et al., 2022, arXiv:2210.07317).
- One flight school's Cessna 172 fleet, 5 years, 1 Hz, 23 channels.
- 28,935 flights, 2,111 unplanned maintenance events, 36 issue types (MaintNet clusters).
- Labels: flights within 2 days before maintenance = "before" (part failing);
  flights after maintenance = "after" (new part). No per-sensor / per-time localisation.
- Benchmark subset: first 5 flights before/after each event, last 4,096 s per flight, 19 classes.
- Paper baselines: ConvMHSA 76.0% binary / 52.8% multiclass; InceptionTime 75.5% / 54.1%.
  Authors report strong overfitting, heavy class imbalance, and that much sensor variance
  is explained by pilot action rather than part condition.
- Aircraft serial numbers and flight dates were removed for privacy: there is no aircraft
  id, event id or date (F1 audits, docs/DATA_CARD.md).

## Status after F5 (2026-10-05)
Built end to end; claims corrected by what the data showed (docs/DECISIONS.md):
- The before/after label separates mostly through oil pressure, which steps at
  maintenance. Outside the oil channels there is little signal, for summaries and for a
  sequence model alike. The system is therefore presented as an evidence-gated support
  prototype, not as failure prediction.
- F3 is a per-flight ridge normal-behaviour model instead of a GRU/LSTM; the deviation
  score has no ranking value on this data and is described as a deviation measure.
- Retrieval and issue-class results are upper bounds and do not beat the frequency
  baseline. The agent says "inconclusive" for about 96% of flights.
- Open: Gemini-written notes, the one-off test-split evaluation, container and AWS (F6).

## Architecture
```
NGAFID ──► Data prep (flight phase · quality · split by flight)
                │
                ▼
     GRU/LSTM normal-behaviour model        ┄┄ evaluated vs InceptionTime baseline
     (trained on healthy flights only)
                │
                ▼
          Residuals per sensor
          │                 │
          ▼                 ▼
 Aircraft health score   Symptom profile
 (fleet ranking)         EGT3 ↑ · CHT3 ↑ · OilP normal
          │                 │
   low ──► monitor          │
   high ───────────────────►┤
                            ▼
              Pinecone: historical maintenance cases
              (built offline from training-set events)
                            │
                            ▼
              ┌──── LangGraph ─────────────────────┐
              │ evidence & confidence gate         │
              │   weak ──► "inconclusive"          │
              │   strong ─► Gemini inspection note │
              │ schema + citation check            │
              └────────────────┬───────────────────┘
                               ▼
                      Technician approval
                               │
                               └──► feedback stored as new case
```
LangGraph orchestrates; Gemini only writes the note from structured evidence.

## MVP scope
In: engine subsystem (CHT1-4, EGT1-4, RPM, OilT, OilP, FFlow); binary task + the 4-5
largest issue classes; normal-behaviour model + trend; similar-case search; agent with
cited note and human approval; FastAPI + Streamlit; one container on AWS.
Out (later): electrical/fuel subsystems, all 36 classes, self-supervised pretraining,
SDR/AD knowledge base, real-time streaming, multi-fleet.

## Phases
F0 setup · F1 data understanding + split (Gate 1: is there signal?) ·
F2 baselines · F3 normal-behaviour model + health score (Gate 2: beats baseline?) ·
F4 similar-case search · F5 agent + API + UI · F6 evaluation + AWS · F7 publish.

## Evaluation (all on the same held-out test split)
| Component | Metric | Compared with |
|---|---|---|
| Health score (flight) | binary accuracy, AUC | InceptionTime, majority class |
| Fleet ranking (aircraft) | precision@k, flights of early warning | random, baseline score |
| Similar cases | top-1 / top-3 issue accuracy (MVP classes) | direct classifier, "always most frequent" |
| Inspection note | unsupported-claim rate; 20 notes hand-checked | — |
| Reliability | inconclusive rate and its correctness | — |
| Detectability | which issue classes leave no sensor trace | hypothesis table |

## Non-negotiables
- No flight in two splits; benchmark folds reused. The data cannot guarantee that an
  event or aircraft stays in one split, so this is stated as a limitation and issue-class
  / retrieval results are reported as upper bounds.
- Case retrieval only sees training-split cases (the data has no dates).
- The LLM never sees raw time series and never invents evidence; weak evidence → "inconclusive".
- Output is decision support, not an airworthiness decision.
