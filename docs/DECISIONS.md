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

## Open
- MVP issue classes (decide in F1 after counts).
- Split unit: aircraft vs. event (decide in F1 after checking the header table).
