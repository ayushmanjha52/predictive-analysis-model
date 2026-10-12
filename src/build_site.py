"""
Builds the static version of the dashboard into site/ for GitHub Pages.

  - Every API response the dashboard reads is rendered once through the
    real FastAPI app (app.py) and written to site/api/*.json, so the
    static site shows exactly what the server would.
  - The deployed classifier (TF-IDF + logistic regression) is exported
    to site/model.json and runs in the browser via frontend/model.js,
    so delay classification and shadow-mode review work without a server.

Run:  python src/build_site.py           then   python src/publish_site.py
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from fastapi.testclient import TestClient
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import config
from data_loader import load_master_events
from feature_engineering import DOMAIN_KEYWORDS, PLANT_TYPES
from predict import EXPLICIT_DEVICE_NAMES, get_predictor

SITE = ROOT / "site"
COUNTRIES = ["", "IND", "USA", "CHN", "JPN", "DEU", "GBR", "FRA", "BRA", "RUS", "AUS"]


def slug(path: str, params: dict) -> str:
    """Must match staticSlug() in frontend/script.js."""
    name = path.strip("/").replace("/", "_") or "root"
    parts = [f"{k}-{params[k]}" for k in sorted(params) if params[k] not in ("", None)]
    return name + ("--" + "_".join(parts) if parts else "")


def export_model() -> dict:
    model = joblib.load(config.MODEL_PATH)
    vec = joblib.load(config.VECTORIZER_PATH)
    meta = joblib.load(config.FEATURE_NAMES_PATH)
    if not hasattr(model, "coef_"):
        raise SystemExit(f"Browser export supports linear models only, got {type(model).__name__}.")

    tfidf = {}
    for name, t in vec.transformer_list:
        vocab = sorted(t.vocabulary_.items(), key=lambda kv: kv[1])
        tfidf[name] = {
            "analyzer": t.analyzer,
            "ngram_range": list(t.ngram_range),
            "vocabulary": [term for term, _ in vocab],
            "idf": [round(float(x), 10) for x in t.idf_],
        }
    predictor = get_predictor()
    durations = {d: {k: (None if v != v else round(float(v), 1)) for k, v in s.items()}
                 for d, s in predictor._device_duration_stats.items()}
    return {
        "classes": [str(c) for c in model.classes_],
        "coef": np.round(model.coef_, 10).tolist(),
        "intercept": np.round(model.intercept_, 10).tolist(),
        "feature_names": meta["feature_names"],
        "known_areas": meta["known_areas"],
        "home_classes": meta.get("home_classes") or [],
        "plant_types": PLANT_TYPES,
        "home_plant_type": config.HOME_PLANT_TYPE,
        "domain_keywords": DOMAIN_KEYWORDS,
        "stop_words": sorted(ENGLISH_STOP_WORDS),
        "tfidf": tfidf,
        "explicit_rule": bool(meta.get("explicit_mention_rule")),
        "explicit_rule_precision": meta.get("explicit_rule_precision"),
        "explicit_names": EXPLICIT_DEVICE_NAMES,
        "confidence_threshold": config.CONFIDENCE_THRESHOLD,
        "durations": durations,
    }


def main():
    from app import app
    client = TestClient(app)

    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "api").mkdir(parents=True)

    events = load_master_events()
    devices = sorted(events["primary_device"].dropna().unique())
    requests = [
        ("/stats", {}), ("/devices", {}), ("/model_info", {}),
        ("/forecast/all_devices", {"months": 6}),
        ("/device_area_breakdown", {"top_n": 12}),
        ("/area_risk", {"top_n": 6}),
        ("/review_queue", {"limit": 30}),
        ("/unresolved_hotspots", {"top_n": 3}),
        ("/health_score", {"top_n": 10}),
        ("/fleet/overview", {}), ("/fleet/lessons", {}), ("/fleet/device_ranking", {}),
        ("/fleet/trend", {}), ("/fleet/grid", {}), ("/fleet/plant_reliability", {}),
    ]
    requests += [("/forecast", {"device": d, "months": 6}) for d in devices]
    requests += [("/power_plants", {"country": c, "top_n": 10}) for c in COUNTRIES]

    for path, params in requests:
        r = client.get(path, params={k: v for k, v in params.items() if v != ""})
        if r.status_code != 200:
            raise SystemExit(f"{path} {params} -> {r.status_code}: {r.text[:200]}")
        (SITE / "api" / f"{slug(path, params)}.json").write_text(json.dumps(r.json()), encoding="utf-8")

    # Shadow-mode baseline (entries already in the repo's log); the
    # browser adds the visitor's own reviews from localStorage.
    log = client.get("/shadow_stats").json()
    (SITE / "api" / "shadow_stats.json").write_text(json.dumps(log), encoding="utf-8")

    (SITE / "model.json").write_text(json.dumps(export_model(), separators=(",", ":")), encoding="utf-8")

    for f in (ROOT / "frontend").iterdir():
        shutil.copy2(f, SITE / f.name)
    html = (SITE / "index.html").read_text(encoding="utf-8")
    html = html.replace('<script src="config.js"></script>',
                        '<script>window.PDM_STATIC = true;</script>\n'
                        '<script src="config.js"></script>\n<script src="model.js"></script>')
    (SITE / "index.html").write_text(html, encoding="utf-8")
    (SITE / ".nojekyll").write_text("", encoding="utf-8")

    n = len(list((SITE / "api").glob("*.json")))
    size = sum(p.stat().st_size for p in SITE.rglob("*") if p.is_file()) / 1e6
    print(f"Built {SITE} -- {n} API snapshots, {size:.1f} MB")


if __name__ == "__main__":
    main()
