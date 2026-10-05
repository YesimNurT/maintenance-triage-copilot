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
- 2026-10-04 · F2.3 result, without oil: no sequence-level signal by the pre-set rule,
  narrowly · InceptionTime on 15 channels, val AUC 0.570 (95% CI 0.546–0.591), accuracy
  54.4% against 52.4% majority; best hold-out AUC 0.576 at epoch 7, stopped at epoch 12,
  training loss 0.698 → 0.681 (chance 0.693). Above chance but weak, and only slightly
  above the per-flight summaries (0.541) · Kaggle T4 run of
  `scripts/f2_sequence_baseline.py --channels without_oil`, 6,271 fit / 697 hold-out /
  2,436 val flights.
- 2026-10-04 · F2.3 result, all three channel sets (same run, same split) · all 17
  channels: val AUC 0.779 (0.760–0.796), accuracy 71.2%, used all 30 epochs; oil only (2
  channels): 0.655 (0.634–0.676), 60.4%, 19 epochs; without oil (15 channels): 0.570
  (0.546–0.591), 54.4%, 12 epochs. Same picture as the per-flight summaries (0.773 /
  0.677 / 0.541): the set-up learns when oil is present, so the weak without-oil result is
  not a training failure.
- 2026-10-04 · Conclusion of F2: the before/after label is mostly separable through oil
  pressure, which steps at maintenance. Without oil channels neither summaries nor a
  sequence model reach the pre-set bar. A health score trained on this label would mainly
  track servicing, not part condition.
- 2026-10-05 · F2.3 was run twice on Kaggle; the files in `results/f2/sequence_*.json`
  are the second run (Save & Run All) · run 2: all 0.773 (0.754–0.791), oil only 0.661
  (0.639–0.681), without oil 0.557 (0.535–0.577); run 1 (interactive, numbers above):
  0.779 / 0.655 / 0.570. Same seed, so GPU training is not bit-reproducible; run-to-run
  difference is about 0.01 AUC and does not change the conclusion (without oil fails the
  pre-set bar in both runs).
- 2026-10-05 · F2.2 set-up and criterion, fixed before seeing results: issue class among
  the five MVP classes from per-flight features, class-weighted logistic regression and
  gradient boosting, with and without oil features, val split. Trained and evaluated
  once on "before" flights and once on "after" flights as a control: after the repair the
  symptom should be gone, so class signal that survives in "after" flights is aircraft /
  period identity (leakage) rather than a fault symptom. Class signal counts as present
  if the macro one-vs-rest val AUC on "before" flights has a 95% CI lower bound above
  0.55, and as symptom-specific only if that lower bound is also above the "after" AUC.
- 2026-10-05 · F2.2 result: weak issue-class signal, mostly not symptom-specific ·
  gradient boosting on "before" flights (2,236 train / 772 val, five MVP classes): macro
  AUC 0.672 (0.642–0.703), balanced accuracy 0.37 (chance 0.20), top-1 35.9% and top-3
  83.5% against 53.8% and 88.2% for "always the most frequent". Control on "after"
  flights: macro AUC 0.631 (0.601–0.661). Oil features make no difference (0.674 without).
  Logistic regression: 0.559 before, 0.561 after · `results/f2/issue_class.json`.
- 2026-10-05 · Reading of F2.2: the pre-set rule is formally met (lower bound 0.642 above
  0.55 and above the control's 0.631), but the rule compared a lower bound with a point
  estimate and the two intervals overlap, so this is not treated as evidence of a fault
  symptom. About three quarters of the lift over 0.5 is still there after the repair,
  which points to aircraft / period identity or class-specific confounds. On top-1 and
  top-3 accuracy the model does not beat the frequency baseline.
- 2026-10-05 · Direction after F2: keep the architecture, correct the claim (user said to
  continue on 2026-10-05). The system is presented as an evidence-gated maintenance
  support prototype, not as failure prediction: the health score is reported with and
  without oil channels, issue-class and retrieval numbers are upper bounds, and the agent
  answers "inconclusive" when evidence is weak. The real test of the method needs
  aircraft ids, dates and maintenance record content, which an operator's data has.
- 2026-10-05 · F3 is kept thin: the normal-behaviour model is a per-flight ridge
  regression that predicts cruise CHT, EGT and oil values from how the flight was flown
  (RPM, fuel flow, phase shares, and OAT / altitude / airspeed once features are rebuilt),
  trained on post-maintenance train flights only. Residuals are scaled by the healthy
  residual spread; the health score is their root mean square. No GRU/LSTM for now ·
  three models already agree that there is little signal outside oil, so a heavier
  normal-behaviour model is not where the effort pays off.
- 2026-10-05 · Gate 2 criterion, fixed before seeing results: the unsupervised health
  score has ranking value if its val AUC (before vs after) has a 95% CI lower bound above
  0.55; reported with and without oil and next to the supervised baselines (0.773 /
  0.541) and precision@k against the before-flight base rate. Under the direction above
  the gate does not block F4–F5; it decides how the score is described.
- 2026-10-05 · Gate 2 result: the unsupervised health score has no ranking value by the
  pre-set rule · val AUC 0.515 (0.491–0.538) with all channels and 0.510 (0.486–0.533)
  without oil; precision@100 0.45 and 0.48 against a 0.476 base rate; 4,722 healthy
  train flights, 9 condition features (no OAT / altitude / airspeed yet) ·
  `results/f3/health_score.json`.
- 2026-10-05 · Diagnostic, not a redefinition of the score: taken one channel at a time
  and with its sign, only the oil-pressure residual separates the groups (AUC 0.387, i.e.
  0.613 for "lower than expected"); CHT residuals 0.52–0.54, EGT 0.48–0.49. The root mean
  square over ten channels ignores direction and dilutes the one channel that carries
  signal. The score definition is left as fixed beforehand; the agent works from the
  per-channel residuals, and the score is described as a deviation measure, not a risk.
- 2026-10-05 · F4 set-up and criterion, fixed before seeing results: a case is a training
  "before" flight of the binary task; its signature is the vector of ten clipped
  residuals; similarity is cosine. Evaluation on the five MVP classes: for each val
  "before" flight take the 10 nearest training cases, rank classes by votes, report top-1
  and top-3 accuracy next to "always the most frequent", and repeat on "after" flights as
  the control. Retrieval counts as useful only if its top-1 accuracy has a 95% bootstrap
  CI lower bound above the frequency baseline. Search runs on a local in-memory index
  with the same interface as the Pinecone index used in deployment.
- 2026-10-05 · F4 result: retrieval does not beat the frequency baseline · "before"
  flights, 2,236 cases / 772 queries, 10 nearest cases: top-1 50.4% (95% CI 46.9–54.3)
  against 53.8% for "always the most frequent"; top-3 92.9% against 88.2%. Control on
  "after" flights: top-1 44.0% against 51.2%, top-3 88.5% against 86.1%. With five
  classes top-3 says little. Consequence for the agent: similar cases are shown as
  context with their vote share, and the evidence gate needs both a strong residual and
  agreeing cases before any check is suggested · `results/f4/retrieval.json`.
- 2026-10-05 · Agent design (F5): the gate and the checks are code, not LLM calls. Gate:
  largest residual |z| ≥ 3.0, at least 3 similar cases with cosine similarity ≥ 0.6, and
  the top issue label holding ≥ 50% of those cases; otherwise the note is "inconclusive"
  and no check is suggested. A drafted note is rejected unless every claim cites known
  evidence ids (symptoms an R id, checks a C id) and quotes only numbers present in the
  evidence. Confidence is set by rule from case agreement, never by the model. Gemini only
  phrases the note; a deterministic template writer is the offline fallback and the test
  double. Thresholds were chosen as round values before looking at outcomes.
- 2026-10-05 · Known limit of the citation check: it proves that a claim points at real
  evidence and invents no numbers, not that the sentence follows from that evidence. That
  part needs the hand check of notes planned for F6.
- 2026-10-05 · F5 result with the template writer on 2,436 val flights: the agent stays
  silent on 96–97% of flights and does not tell the groups apart · note rate 3.0% on
  "before" flights (35 notes) and 4.0% on "after" flights (51 notes); every produced note
  passes the citation check; the suggested issue matches the logged one in 49% and 45% of
  notes. Main stop reasons: residuals within normal range, similar cases disagree ·
  `results/f5/agent_eval.json`. Gemini-written notes not evaluated yet (needs an API key).
- 2026-10-05 · CI now installs the agent, api and app extras (still no torch) · the
  agent, API and Streamlit tests would otherwise be skipped in CI; replaces the
  2026-10-02 "core + dev only" choice.
- 2026-10-05 · F3–F5 rerun after rebuilding the features with cruise OAT, altitude and
  airspeed (12 condition features instead of 9); these numbers replace the first run, the
  conclusions do not change · F3: score AUC 0.519 (0.497–0.542) with all channels, 0.507
  (0.485–0.530) without oil, precision@100 0.48 / 0.46; signed oil-pressure residual 0.38
  (0.62 for "lower than expected"). F4: top-1 50.1% (46.9–53.6) against 53.8%, top-3 93.1%
  against 88.2%; control 44.8% against 51.2%. F5: note rate 3.8% on "before" flights (44
  notes) and 4.2% on "after" flights (53 notes), citation validity 100%, suggested issue
  matches the logged one in 41% and 47% of notes. Gate 1 and F2 tabular results are
  identical to before, as the context features are not part of the engine feature set.

## Open
- Gemini-written notes: run `f5_agent_eval.py --llm --limit 40` with an API key, hand-check
  20 notes.
- F6: one-off evaluation on the test split, container, AWS. Not started.
