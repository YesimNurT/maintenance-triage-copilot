# Decisions log

Format: date · decision · why · evidence / number.

- 2026-09-24 · Rejected ESA-ADB (ESA Anomaly Dataset) · channels, timelines and measured
  value types are anonymised by design, so no defensible link between an anomaly and
  external maintenance knowledge is possible · ESA-ADB paper (arXiv:2406.17826).
- 2026-09-25 · Dataset: NGAFID maintenance dataset · named sensors, real maintenance
  labels (36 issue types), public (Zenodo / Kaggle) · arXiv:2210.07317.
- 2026-09-25 · Scope: triage copilot (fleet ranking + similar-case retrieval + cited LLM
  note). Fault localisation is shown as explanation only, not claimed, because the data
  has no location labels.
- 2026-09-25 · LLM: Gemini via langchain-google-genai; orchestration: LangGraph.
- 2026-10-02 · Download source: Zenodo record 6624956 (not Kaggle) · plain HTTPS, no
  credentials, md5 published per file so the download is verifiable · files:
  `all_flight.tar.gz` 4.29 GB, `2days.tar.gz` 1.13 GB (Zenodo record API).
- 2026-10-02 · CI installs core + dev dependencies only (no torch / agent extras) · keeps
  the run fast; tests must mock Gemini and Pinecone anyway.
- 2026-10-03 · Split unit: flight. Benchmark flights keep their published fold (5 folds);
  other flights get a fold by seeded random assignment stratified by label and
  before/after. Fold 4 = test, fold 3 = val, folds 0–2 = train · the data has no aircraft
  or event id (removed for privacy), row-order reconstruction gives 9,387 groups against
  2,111 events, and neighbouring flights are no more alike than random ones (OAT median
  difference 12.29 lag 1 vs 12.26 random), so no split can keep an event together;
  reusing the folds keeps results comparable with the paper · `results/f1/*_audit.json`.
- 2026-10-03 · Known limitation, stated in every result: flights of the same aircraft or
  event can fall in different splits. Risk is lower for before/after (an event's flights
  carry both labels) and higher for issue-class accuracy and case retrieval, which are
  reported as upper bounds.
- 2026-10-03 · Case retrieval sees only training-split cases (replaces "only cases dated
  before the query": the data has no dates).
- 2026-10-03 · Binary task definition (Gate 1 and F2): benchmark rule, before = −2 ≤
  date_diff ≤ −1, after = 1 ≤ date_diff ≤ 2; "same" and day-0 flights excluded.
- 2026-10-03 · Gate 1 criterion, fixed before seeing results: logistic regression on
  per-flight summary features (train split) passes if the lower bound of the 95%
  bootstrap CI of val AUC is above 0.55 and above the flight-length-only model. Test
  split stays untouched until F6.
- 2026-10-03 · Provisional (to be checked with `scripts/audit_channels.py`): plausible
  value ranges per channel in `src/mtc/data/quality.py`, flight minimum 600 s, phase rules
  in `src/mtc/data/phases.py`.

## Open
- MVP issue classes: decide after Gate 1, from per-label val AUC (which classes leave a
  sensor trace). Event counts per class are not recoverable.
