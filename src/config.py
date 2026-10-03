"""Central configuration -- every path/constant used anywhere in this
project is defined here ONCE. Every other module imports from here;
nothing hardcodes a path or a month order independently. This is the
single fix that prevents the recurring 'two files disagree about X'
class of bug found repeatedly earlier in this project.
"""
import os
from datetime import datetime
from pathlib import Path

# Resolved relative to this file, not hardcoded to one machine's path --
# works regardless of where the project folder is cloned/moved to.
PROJECT_ROOT = Path(os.environ.get("PDM_ROOT", Path(__file__).resolve().parent.parent))

REPO_DATA_DIR = PROJECT_ROOT / "data"
# PDM_DATA_DIR: put the WRITABLE data (event log, shadow log) somewhere
# persistent, e.g. a Render disk. On first start the directory is seeded
# from the repo's data/ folder, then left alone so logged events survive
# redeploys. Unset = use the repo's data/ folder (local dev).
DATA_DIR = Path(os.environ.get("PDM_DATA_DIR", REPO_DATA_DIR))
EXTERNAL_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, EXTERNAL_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

if DATA_DIR.resolve() != REPO_DATA_DIR.resolve() and REPO_DATA_DIR.exists():
    import shutil
    for _src in REPO_DATA_DIR.rglob("*"):
        _dst = DATA_DIR / _src.relative_to(REPO_DATA_DIR)
        if _src.is_file() and (not _dst.exists() or _dst.parent.name == "external"):
            # external/ is public reference data shipped with each deploy
            # -- always refreshed; everything else is seeded only once.
            _dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_src, _dst)

RAW_DELAYS_PATH = DATA_DIR / "raw_delays.xlsx"
LEGACY_MASTER_PATH = DATA_DIR / "legacy_master_events.csv"
MASTER_EVENTS_PATH = DATA_DIR / "master_events.csv"
FMEA_SCORES_PATH = DATA_DIR / "fmea_risk_scores.csv"

# Public power-sector data (see src/external_data.py). Kept in a
# separate file from master_events.csv: master_events.csv is the
# Combi Mill's own hand-tagged log (and what /log_event appends to);
# external events are weakly-labeled public data and are never used to
# measure accuracy.
EXTERNAL_EVENTS_PATH = EXTERNAL_DIR / "external_events.csv.gz"
WRI_PLANTS_PATH = EXTERNAL_DIR / "global_power_plants.csv.gz"

# The site whose own hand-tagged data is the honest holdout benchmark.
HOME_SITE = "COMBI_MILL"
HOME_PLANT_TYPE = "STEEL_ROLLING_MILL"

MODEL_PATH = MODELS_DIR / "cause_classifier_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
MANIFEST_PATH = MODELS_DIR / "training_manifest.json"

CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
HOLDOUT_REPORT_PATH = REPORTS_DIR / "holdout_report.csv"

# Shadow-mode validation: records model prediction + human's independent
# actual diagnosis for the SAME real delay, so agreement rate can be
# measured before the model drives unsupervised decisions. Deliberately
# a separate file from master_events.csv -- shadow entries are not
# training data until reviewed and promoted.
SHADOW_LOG_PATH = DATA_DIR / "shadow_log.csv"

# Devices with fewer than this many labeled examples are dropped from
# training -- not enough signal for k-fold CV or a stable classifier.
MIN_CLASS_COUNT = 5

# Below this predicted probability, a prediction is flagged
# low-confidence rather than presented as a normal result.
CONFIDENCE_THRESHOLD = 0.35

# Chronological order -- month label strings like "Nov-25" sort WRONG
# alphabetically ("Apr-26" < "Dec-25"). Always order via
# month_sort_key(), never via .sort_values() on the raw string.
#
# FIX: this used to be a hardcoded list (Nov-25..May-26). Any month
# outside it -- including every event /log_event writes from Jun-26
# onward -- sorted as "unknown" and was silently DROPPED from the
# forecast series. Months are now parsed, so the order never goes stale.
MONTH_ORDER = ["Nov-25", "Dec-25", "Jan-26", "Feb-26", "Mar-26", "Apr-26", "May-26"]  # kept for reference only


def month_sort_key(month_label) -> int:
    """'Nov-25' -> 202511. Unparseable labels sort last, never crash."""
    try:
        d = datetime.strptime(str(month_label), "%b-%y")
        return d.year * 100 + d.month
    except ValueError:
        return 999999


def sort_months(labels) -> list:
    return sorted({m for m in labels if isinstance(m, str)}, key=month_sort_key)


# The canonical set of field devices this project tracks for headline
# KPIs/dashboard framing. Training itself uses whatever classes have
# >= MIN_CLASS_COUNT examples (may include more or fewer than this list
# depending on the data) -- this list is for reporting/dashboard
# emphasis only, never used to filter training data.
TARGET_DEVICES = [
    "PHOTOCELL", "HMD", "PROXIMITY", "LVDT", "ENCODER",
    "PRESSURE_SWITCH", "FLOW_SWITCH", "LASER",
]

CORS_ALLOW_ORIGINS = os.environ.get("PDM_CORS_ORIGINS", "").split(",") if os.environ.get("PDM_CORS_ORIGINS") else [
    "http://127.0.0.1:5500", "http://localhost:5500",  # local dev (Live Server default port)
]
# PRODUCTION: set the PDM_CORS_ORIGINS environment variable to the real
# server's address before deploying, e.g.:
#   PDM_CORS_ORIGINS=http://10.0.4.12:8080,http://plant-pdm.local:8080
# Never leave this as "*" once real maintenance decisions depend on this
# app -- an open CORS policy lets any website make requests to this API
# on a plant network. The dev defaults above only work for local testing.