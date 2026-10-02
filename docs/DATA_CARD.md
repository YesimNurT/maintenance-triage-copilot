# Data card — NGAFID maintenance dataset

Numbers below come from `scripts/audit_header.py` (2026-10-02), report in
`results/f1/header_audit.json`.

## Source and version
Zenodo record 6624956 (Yang et al., 2022, arXiv:2210.07317), downloaded with
`scripts/download_ngafid.py`, md5 verified.

| Path under `data/raw/ngafid/` | Content |
|---|---|
| `all_flights/flight_header.csv` | 28,935 flights, one row per flight |
| `all_flights/one_parq/` | 1 Hz sensor rows, 401 parquet files, 5.96 GB |
| `2days/flight_header.csv` | benchmark subset, 11,446 flights |
| `2days/flight_data.pkl` | sensor data of the subset (not opened yet) |
| `2days/stats.csv` | per-channel max / min over the subset |

## Header table columns and meaning
`all_flights/flight_header.csv` (no missing values except `hierarchy`, no duplicate rows):

| Column | Values | Meaning |
|---|---|---|
| `Master Index` | 28,935 unique, range 1–32,820 (gaps) | flight id, join key to the parquet rows |
| `before_after` | before 12,140 · after 10,291 · same 6,504 | position relative to the maintenance event |
| `date_diff` | −108 … 70 | days from the maintenance date |
| `flight_length` | 12 … 30,059 | seconds |
| `label` | 36 issue types | MaintNet cluster of the maintenance record |
| `hierarchy` | baffle 5,451 · engine 2,959 · cylinder 1,578 · oil 593 · missing 18,354 | coarse group, absent for 63% of flights |
| `number_flights_before` | 0: 4,009 · 1: 2,038 · 2: 2,035 · 3: 2,031 · 4: 2,027 · −1: 16,795 | set for "before" flights only; −1 for "after" and "same" |

The benchmark header adds `fold` (5 equal folds), `class` (1–19), `target_class`
(0 = after, otherwise the class) and `hclass`; there `before_after` is 1 / 0
(5,602 before, 5,844 after) and `date_diff` is limited to −2, −1, 1, 2.

Sensor columns (parquet): `volt1 volt2 amp1 amp2 FQtyL FQtyR` · `E1 FFlow` `E1 OilT`
`E1 OilP` `E1 RPM` `E1 CHT1-4` `E1 EGT1-4` · `OAT IAS VSpd NormAc AltMSL` (23 channels),
plus `timestep`, `cluster`, `Master Index`. All 12 MVP engine channels are present.

**There is no aircraft id and no maintenance-event id**, in the header or in the parquet
schema. The paper states that aircraft serial numbers and flight dates were removed for
privacy. The parquet `cluster` column is the issue label under another name: 36 values,
one per flight, one-to-one with `label`; every flight in the parquet is in the header.

Event reconstruction attempt (`scripts/audit_events.py`, 2026-10-02): starting a new group
at every label change or backwards phase step gives 9,387 groups, not the 2,111 events of
the paper; 685 groups repeat a `number_flights_before` value. Median group size is 3
flights. Events cannot be recovered from row order with this rule.

"before" flights are the last flights before maintenance whatever the gap in days
(`date_diff` down to −108); `number_flights_before` 1–4 occur about 2,030 times each and
0 occurs 4,009 times, including flights on the maintenance day itself.

## Aircraft / events / flights per class
Aircraft and event counts cannot be read from the header (no ids). Flights per label,
all flights:

| Label | Flights | Share |
|---|---|---|
| intake gasket leak/damage | 10,140 | 35.0% |
| rocker cover leak/loose/damage | 5,219 | 18.0% |
| intake tube/bolt/seal/boot loose or damage | 1,372 | 4.7% |
| baffle crack/damage/loose/miss | 1,294 | 4.5% |
| baffle plug need repair/replace | 1,142 | 3.9% |
| baffle screw miss/loose | 894 | 3.1% |
| baffle seal loose/damage | 859 | 3.0% |
| engine run rough | 670 | 2.3% |
| baffle tie/tie rod loose or damage | 656 | 2.3% |
| engine failure/fire/time out | 643 | 2.2% |
| remaining 26 labels | 6,046 | 20.9% |

## Quality issues found and how they are handled
Found so far (handling to be decided in the loading / quality task):
- 6,504 "same"-day flights: neither clearly before nor after the repair.
- Very short recordings (minimum 12 s) and very long ones (30,059 s).
- `2days/stats.csv` shows physically impossible minima (CHT −304, EGT −336, OilT −58,
  RPM 0), so the sensor data contains invalid readings.
- `2days/stats.csv` is computed over the whole subset, so it must not be used for
  normalisation (statistics come from the training split only).

## Flight-phase rules
To be filled in F1.

## Split rule and resulting class distribution
To be filled in F1.
