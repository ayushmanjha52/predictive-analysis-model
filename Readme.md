# Combi Mill Field-Device Predictive Maintenance

Supports the FY27 target: Reliability Enhancement of field devices by
reduction of delays by 50% in FY27 over FY26.

## Documentation map
- **This file** -- setup and day-to-day commands
- **DEPLOYMENT.md** -- production server setup (CORS, firewall, NSSM/systemd)
- **MODEL_CARD.md** -- current model, real accuracy numbers, known limitations
- **LOGGING_GUIDE.md** -- how to log delays so more auto-tag correctly

## Setup
```
pip install -r requirements.txt
```

## Rebuild data from scratch (if you get new raw files)
```
python src/ingest.py
```
Reconciles data/raw_delays.xlsx (primary) with data/legacy_master_events.csv
(supplementary rows only, deduplicated by reason text) into data/master_events.csv.

## Train
```
python src/train.py
```
Compares Random Forest, HistGradientBoosting, Logistic Regression (+ CatBoost
if installed). Selects the winner by HONEST TIME-BASED HOLDOUT accuracy
(train on all-but-latest month, test only on that unseen month) -- not
cross-validation score alone. Also prints the top device x area combinations.
See MODEL_CARD.md for current production numbers -- regenerate that file's
numbers after every retrain.

## Test
```
python tests/test_all.py
```
10 tests, all currently passing against the real trained model.

## Run the API (development)
```
uvicorn app:app --reload --port 8000
```
For production deployment (always-on service, real network access,
correct CORS/firewall setup), see DEPLOYMENT.md instead -- do not use
`--reload` in production.

## Top device x area combinations (from reports/device_area_breakdown.csv)
1. PHOTOCELL @ SAW -- 887 min (37 events)
2. LVDT @ BDM -- 872 min (17 events)
3. HMD @ BDM -- 843 min (41 events)
4. PHOTOCELL @ COOLING_BED -- 585 min (26 events)
5. ENCODER @ BDM -- 483 min (10 events)

## Data reconciliation note
data/raw_delays.xlsx and the legacy master_events.csv were confirmed to be
95% overlapping exports of the same underlying delays. src/ingest.py
reconciles them without double-counting -- see the module docstring for
full detail. One known open question: two rows in raw_delays.xlsx for
"1 billet rejected at BDM due to HMD Continuous sensing" on 18-Nov-2025 are
byte-for-byte identical (same 20 min duration) -- worth checking the
original spreadsheet to confirm this is a genuine duplicate entry vs. a
copy-paste artifact.

## Shadow-mode validation
The dashboard's "Shadow-Mode Review" panel lets a reviewer log a real
delay, independently diagnose it, then compare against the model's
(initially hidden) prediction. See MODEL_CARD.md's "Validation status"
section -- this has zero resolved entries so far; the model should not
be treated as authoritative for unsupervised decisions until this
accumulates real data.

## Known limitations
See MODEL_CARD.md for the full, current list (accuracy numbers, class
imbalance, Encoder's known weak spot, data coverage gap, validation
status). Summary:
- 344 labeled training events across 8 classes -- a modest sample size.
- Text-only classification -- no live sensor/PLC signal integration.
- Not yet validated in shadow mode against real plant outcomes.