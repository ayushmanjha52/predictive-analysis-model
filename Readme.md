# Field-Device Reliability: Industry Mill + Power Fleet

**Live project:** https://ayushmanjha52.github.io/predictive-analysis-model/
**Source:** https://github.com/ayushmanjha52/predictive-analysis-model

Why do field devices stop the mill, and what should be fixed first?
This project turns the Industry Mill's delay log into a
prioritised reliability picture. It benchmarks those findings against
every reported event at US power reactors (2012 onward) and 35,000
power plants worldwide.

It supports the FY27 target: reduce field-device delays by 50% versus FY26.

## What's in the dashboard
- **01 Industry Mill**: device Pareto, red-zone areas, a 6-month delay
  forecast, recurrence health scores, FMEA risk fusion, a review queue
  for untagged delays, and shadow-mode validation of the classifier.
- **02 Power Fleet**: about 4,800 US NRC power-reactor event reports. It
  shows which field devices fail, the failure modes mined from
  plant-written narratives, trip rates, grid/offsite-power events,
  plants ranked by trips per GW, and lessons for the mill's own devices.
- **03 Global Plants**: installed capacity by fuel and country (WRI
  Global Power Plant Database), including India.

## The classifier
A TF-IDF + logistic-regression model predicts which device (Photocell,
HMD, Proximity, LVDT, Encoder, Pressure/Flow Switch, Laser) caused a delay
from its free-text description. It is selected by a rolling time-based
backtest on the mill's own data: **62.7% across 161 unseen events, 75.0%
on the latest month**. See [MODEL_CARD.md](MODEL_CARD.md) for what was
tried, what helped and the known weak spots.

On the live site the model runs in the browser (`frontend/model.js`, an
exact port of the Python predictor, checked by the test suite), so the
site needs no server.

## Project layout
```
app.py                  FastAPI backend + dashboard server
frontend/               dashboard (HTML/CSS/JS) and in-browser model
src/ingest.py           build data/master_events.csv from raw mill files
src/external_data.py    fetch public NRC + WRI power-plant data (resumable)
src/train.py            model comparison + time-based backtest
src/build_site.py       render the static site into site/
src/publish_site.py     push site/ to the gh-pages branch
tests/test_all.py       data, model, API and browser-parity tests
```

## Run it locally
```
pip install -r requirements-dev.txt
uvicorn app:app --reload --port 8000      # http://127.0.0.1:8000  (API docs at /docs)
```

## Common tasks
```
python src/external_data.py               # refresh power-plant data
python src/train.py                       # retrain (or: python retrain_model.py, with rollback)
python -m pytest tests/test_all.py        # tests
python src/build_site.py && python src/publish_site.py   # update the live site
```
See [DEPLOYMENT.md](DEPLOYMENT.md) for hosting details.

## Data sources
- Industry Mill delay log (internal).
- US NRC Event Notification Reports: public records. Device labels on
  these are keyword-derived and marked as such in the dashboard.
- WRI Global Power Plant Database v1.3: CC BY 4.0, © World Resources Institute.

## Limitations
- 344 hand-tagged mill events across 8 device classes. More tagged mill
  data is the main lever for accuracy.
- Encoder and LVDT are the weakest and most-confused classes.
- Text-only classification, with no live sensor/PLC signals.
