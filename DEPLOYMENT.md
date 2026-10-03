# Deployment

The API and the dashboard are **one service**: `app.py` serves the
dashboard (`frontend/`) at `/` and the JSON API on the other routes.
The browser calls the API on the same origin, so no URL or CORS
editing is needed.

## Render (recommended)

1. Push this repo to GitHub (already wired: `render.yaml` is at the repo root).
2. In the Render dashboard: **New → Blueprint**, then select the repository.
   Render reads `render.yaml` and creates the `field-device-pdm` web service:
   - build: `pip install -r requirements.txt`
   - start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - health check: `/health`
   - Python 3.13.1 (pinned via `PYTHON_VERSION`, matching the version the
     models were trained with)
3. When the deploy finishes, open `https://<service-name>.onrender.com/`.
   The API docs are at `/docs`.

Every push to `main` redeploys automatically (`autoDeploy: true`).

### Things to know about the free plan
- **The service sleeps after ~15 min idle**, and the first request after
  that takes about a minute to wake it.
- **The filesystem is ephemeral.** Events written through `/log_event`
  and shadow-mode entries (`/shadow_predict`, `/shadow_resolve`) are
  stored in CSV files. On Render they are **lost on every redeploy or
  restart**. For real shadow-mode validation, do one of these:
  - attach a Render persistent disk (paid plan) mounted at e.g.
    `/var/data`, and set the env var `PDM_DATA_DIR=/var/data`. On first
    start the app seeds the disk from the repo's `data/` folder. After
    that, logged events and shadow entries survive redeploys. The public
    reference data in `data/external/` is refreshed from the repo on
    every deploy; or
  - run shadow mode on an always-on plant server instead (below).
- The model is **not** retrained on Render. Train locally, commit
  `models/`, and push.

## Plant network server (alternative)

```
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Run it as a service (NSSM on Windows, systemd on Linux) so it restarts
on reboot. Open port 8000 in the firewall. If a frontend on a
*different* origin needs the API, set
`PDM_CORS_ORIGINS=http://that-host:port` (never `*`).

## Updating the deployed model
```
python src/external_data.py   # optional: refresh public power-plant data
python src/train.py           # or: python retrain_model.py (adds a regression guard)
python -m pytest tests/test_all.py
git add models/ reports/ data/external/external_events.csv.gz && git commit && git push
```
`requirements.txt` pins scikit-learn, CatBoost and numpy to the versions
the committed model files were trained with. If you upgrade any of them
locally, retrain before pushing.
