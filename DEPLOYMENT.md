# Deployment

## Public site: GitHub Pages (free, always on)

Live at https://ayushmanjha52.github.io/predictive-analysis-model/

The public site is static, so there is no server to keep running:
- `src/build_site.py` renders every dashboard view through the real
  FastAPI app into `site/api/*.json`. It also exports the classifier to
  `site/model.json`, which `frontend/model.js` runs in the browser.
- `src/publish_site.py` pushes `site/` to the `gh-pages` branch, which
  GitHub Pages serves.

To update the site after new data or a retrain:
```
python src/build_site.py
python src/publish_site.py
```
Pages refreshes within a minute or two. In the repo settings,
**Settings → Pages** should show source "Deploy from a branch", with
`gh-pages` and `/ (root)`.

On the static site, shadow-mode reviews are saved in each reviewer's own
browser (localStorage), not in a shared log.

The in-browser model supports linear models (the current logistic
regression). If a retrain selects a tree model, `build_site.py` stops
with a clear message.

## Plant network server (full backend)

For a shared shadow-mode log and `/log_event` writes, run the API:
```
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```
Run it as a service (NSSM on Windows, systemd on Linux) so it restarts
on reboot, and open port 8000 in the firewall. `app.py` serves the
dashboard at `/` and the API docs at `/docs`. Set `PDM_DATA_DIR` to keep
the writable logs on a persistent path. If a frontend on a different
origin needs the API, set `PDM_CORS_ORIGINS=http://that-host:port`
(never `*`).

`requirements.txt` pins scikit-learn and numpy to the versions the
committed model files were trained with. Retrain before upgrading them.
