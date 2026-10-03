# Field-Device Predictive Maintenance: Combi Mill + Power Fleet

Supports the FY27 target: Reliability Enhancement of field devices by
reduction of delays by 50% in FY27 over FY26.

The dashboard has three tabs:
- **Combi Mill**: the mill's own delay log. It shows device Pareto, area
  red zones, a 6-month forecast, recurrence health scores, FMEA fusion,
  the review queue, and shadow-mode validation.
- **Power Fleet Insights**: every reportable event at US power reactors
  (public NRC Event Notification Reports, 2012 onward). It shows which
  field devices fail, the real failure modes mined from plant-written
  narratives, trip rates, grid/offsite-power events, plants ranked by
  trips per GW, and cross-site lessons for the mill's own devices.
- **Global Power Plants**: the installed base of about 35,000 plants
  worldwide (WRI Global Power Plant Database), by fuel and country.

## Documentation map
- **This file**: setup and day-to-day commands
- **DEPLOYMENT.md**: Render deployment (and plant-server alternative)
- **MODEL_CARD.md**: current model, honest accuracy, what helped and what didn't

## Setup
```
pip install -r requirements-dev.txt    # requirements.txt = runtime only (what Render installs)
```

## Data
```
python src/ingest.py          # rebuild data/master_events.csv from raw mill files
python src/external_data.py   # fetch/refresh public NRC + WRI power-plant data (resumable)
python src/external_data.py --relabel   # re-run device/failure-mode tagging on cached NRC data
```
nrc.gov rate-limits bursts. The fetcher is polite (default 2 workers),
backs off on 403s, and resumes where it stopped, so it is safe to re-run.

## Train
```
python src/train.py           # or: python retrain_model.py (adds rollback + regression guard)
```
Compares model families × data mixes (mill-only vs. + power-plant data)
× an explicit-device-mention rule. It selects by an **honest rolling
backtest** on the mill's own last 3 months (train strictly before, test
on the unseen month). Power-plant data never enters a test set and is
only adopted if it wins by at least 2 points. See MODEL_CARD.md.

## Test
```
python -m pytest tests/test_all.py
```

## Run locally
```
uvicorn app:app --reload --port 8000
```
Open http://127.0.0.1:8000/ for the dashboard and /docs for the API.

## Deploy
Render Blueprint (`render.yaml`): New → Blueprint → select this repo.
See DEPLOYMENT.md.

## Data sources & licences
- Combi Mill delay log: internal.
- US NRC Event Notification Reports: public US-government records.
  Device labels on these are keyword-derived (weak labels), and the
  API and dashboard say so.
- WRI Global Power Plant Database v1.3: CC BY 4.0, © World Resources Institute.

## Known limitations
See MODEL_CARD.md. Summary:
- 344 labeled mill events across 8 classes. More hand-tagged mill data
  is the main lever for accuracy.
- Encoder/LVDT are the weakest, most-confused classes.
- Text-only classification, with no live sensor/PLC signals.
- Not yet validated in shadow mode at scale.
