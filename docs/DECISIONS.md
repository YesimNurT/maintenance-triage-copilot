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
- 2026-10-03 · Value ranges per channel (`src/mtc/data/quality.py`) confirmed; lower
  bounds of OilP and IAS widened to −5 for noise around zero · on 300 random flights
  (1.15 M rows) at most 0.2% of a channel was outside (OilP, slightly negative readings);
  medians CHT 294–307 °F, EGT 1277–1291 °F, RPM 2225 · `results/f1/channel_audit.json`.
- 2026-10-03 · Minimum flight length 600 s kept · the benchmark's shortest flight is
  605 s, so this matches the paper; 25% of all flights are shorter than 415 s (probably ground
  runs, aborted recordings) and 208 of 300 sampled flights pass length + missing checks.
- 2026-10-03 · Phase rules (`src/mtc/data/phases.py`) kept · row shares on the sample:
  cruise 47%, ground 28%, descent 13%, climb 12%, none unknown. "Cruise" means level
  airborne flight and includes traffic-pattern legs.
- 2026-10-03 · Gate 1 passed by the pre-set rule · val AUC 0.773 (95% CI 0.755–0.792),
  accuracy 71.0% against 52.4% majority, length-only AUC 0.524; 6,968 train / 2,436 val
  flights after dropping 2,399 train/val flights for length or missing values · `results/f1/gate1_signal.json`.
- 2026-10-03 · Caveat on Gate 1, to be carried into every later result: the signal sits
  in oil pressure. Oil features alone give AUC 0.677, everything except oil 0.541,
  cylinder-relative features (the "EGT3 up" kind of symptom) 0.518. Median cruise oil
  pressure is 70.1 psi after against 68.9 psi before maintenance. Working hypothesis, not
  verified: maintenance visits include oil servicing, so the model partly detects "recently
  serviced" rather than "part failing". From F2 on, every model is reported with and
  without oil channels.
- 2026-10-03 · Per-label val AUC (labels with at least 20 flights per class): baffle plug
  0.89, intake tube 0.87, intake gasket 0.84, baffle tie 0.82, baffle crack 0.82, rocker
  cover 0.80, baffle screw 0.80, cylinder compression 0.74, baffle seal 0.72; engine
  failure 0.36 and engine run rough 0.35 (below 0.5: the pattern is reversed there).
- 2026-10-04 · MVP issue classes: intake gasket leak/damage, rocker cover
  leak/loose/damage, baffle crack/damage/loose/miss, intake tube/bolt/seal/boot loose or
  damage, baffle plug need repair/replace · the five largest classes in the val binary
  task (862, 446, 119, 116, 102 flights); chosen by size, not by AUC, because per-label
  AUC mostly reflects the oil-pressure effect.
- 2026-10-04 · F2 starts with the oil hypothesis: first baseline is built with and
  without oil channels before any sequence model is trained.
- 2026-10-04 · F2.1 criterion, fixed before seeing results: CHT/EGT signal counts as
  present if a model without oil features reaches a val AUC whose 95% bootstrap CI lower
  bound is above 0.55 (logistic regression or gradient boosting on per-flight features).
- 2026-10-04 · F2.1 result: no CHT/EGT signal in per-flight summaries by the pre-set rule
  · without oil features val AUC is 0.541 (CI 0.520–0.564) for logistic regression and
  0.539 (0.516–0.563) for gradient boosting; oil only 0.677 / 0.684; all features 0.773 /
  0.741 · `results/f2/tabular_baselines.json`.
- 2026-10-04 · Oil pressure steps at maintenance, it does not drift · median cruise oil
  pressure 68.97 / 68.83 psi at days −2 / −1 and 70.04 / 70.09 psi at days +1 / +2 (9,404
  train/val flights). Per label the step is +1.0 to +1.6 psi for gasket, rocker cover,
  baffle and intake labels and about zero or negative for engine run rough, engine
  failure, idle/rpm, start and pilot-noticed labels · consistent with the hypothesis that
  some maintenance visits include oil servicing; still not verified against maintenance
  records · `results/f2/oil_hypothesis.json`.
- 2026-10-04 · Oil pressure over a wider window confirms a step, not a drift · median
  cruise oil pressure stays at 68.8–69.2 psi from day −6 to −1, is 69.3 on day 0 and
  70.0–70.7 psi from day +1 to +6 (15,425 usable train/val flights, days with at least 20
  flights) · `results/f2/oil_hypothesis.json`.
- 2026-10-04 · F2.3 set-up and criterion, fixed before seeing results: InceptionTime on
  the last 4,096 s at 1 Hz, 12 engine + 5 context channels (not the paper's 23), trained on
  a Kaggle GPU via `notebooks/kaggle_f2_sequence.ipynb` (the Mac has 8 GB RAM; one MPS
  step of 32 flights took 1.05 s and 3.4 GB); early stopping on 10% of train, val reported, test untouched. Three
  channel sets: all, without oil (OilT, OilP removed), oil only. Sequence-level signal
  outside oil counts as present if the without-oil val AUC has a 95% CI lower bound
  above 0.55.

## Open
- Does a sequence model on 1 Hz data find before/after signal without oil channels
  (F2.3)? Per-flight summaries do not. If it does not either, the before/after label
  mostly marks "recently serviced" and the health score scope is revisited.
