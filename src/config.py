"""Central configuration -- every path/constant used anywhere in this
project is defined here ONCE. Every other module imports from here;
nothing hardcodes a path or a month order independently. This is the
single fix that prevents the recurring 'two files disagree about X'
class of bug found repeatedly earlier in this project.
"""
import os
from pathlib import Path

# Resolved relative to this file, not hardcoded to one machine's path --
# works regardless of where the project folder is cloned/moved to.
PROJECT_ROOT = Path(os.environ.get("PDM_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_DELAYS_PATH = DATA_DIR / "raw_delays.xlsx"
LEGACY_MASTER_PATH = DATA_DIR / "legacy_master_events.csv"
MASTER_EVENTS_PATH = DATA_DIR / "master_events.csv"
FMEA_SCORES_PATH = DATA_DIR / "fmea_risk_scores.csv"

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
# alphabetically ("Apr-26" < "Dec-25"). Always order via this list,
# never via .sort_values() on the raw string.
MONTH_ORDER = ["Nov-25", "Dec-25", "Jan-26", "Feb-26", "Mar-26", "Apr-26", "May-26"]


def month_sort_key(month_label: str) -> int:
    try:
        return MONTH_ORDER.index(month_label)
    except ValueError:
        return len(MONTH_ORDER)  # unknown months sort last, not crash


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