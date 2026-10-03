"""Central configuration: every path and constant used by the project."""
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PDM_ROOT", Path(__file__).resolve().parent.parent))

REPO_DATA_DIR = PROJECT_ROOT / "data"
# PDM_DATA_DIR: optional persistent location for writable data (event
# log, shadow log), e.g. a Render disk. It is seeded from the repo's
# data/ folder on first start.
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
            # external/ is refreshed on every deploy; the rest is seeded once.
            _dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(_src, _dst)

RAW_DELAYS_PATH = DATA_DIR / "raw_delays.xlsx"
LEGACY_MASTER_PATH = DATA_DIR / "legacy_master_events.csv"
MASTER_EVENTS_PATH = DATA_DIR / "master_events.csv"
FMEA_SCORES_PATH = DATA_DIR / "fmea_risk_scores.csv"

# Public power-sector data (see src/external_data.py), kept separate
# from the mill's own hand-tagged log. Not used to measure accuracy.
EXTERNAL_EVENTS_PATH = EXTERNAL_DIR / "external_events.csv.gz"
WRI_PLANTS_PATH = EXTERNAL_DIR / "global_power_plants.csv.gz"

# The site whose hand-tagged data is the accuracy benchmark.
HOME_SITE = "COMBI_MILL"
HOME_PLANT_TYPE = "STEEL_ROLLING_MILL"

MODEL_PATH = MODELS_DIR / "cause_classifier_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"
MANIFEST_PATH = MODELS_DIR / "training_manifest.json"

CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
HOLDOUT_REPORT_PATH = REPORTS_DIR / "holdout_report.csv"

# Shadow-mode validation log: model prediction + reviewer's independent
# diagnosis for the same delay. Not used as training data.
SHADOW_LOG_PATH = DATA_DIR / "shadow_log.csv"

# Devices with fewer labeled examples than this are excluded from training.
MIN_CLASS_COUNT = 5

# Predictions below this probability are flagged low-confidence.
CONFIDENCE_THRESHOLD = 0.35

# Month labels like "Nov-25" don't sort correctly as strings; use these.
def month_sort_key(month_label) -> int:
    """'Nov-25' -> 202511. Unparseable labels sort last."""
    try:
        d = datetime.strptime(str(month_label), "%b-%y")
        return d.year * 100 + d.month
    except ValueError:
        return 999999


def sort_months(labels) -> list:
    return sorted({m for m in labels if isinstance(m, str)}, key=month_sort_key)


# Field devices offered in the dashboard's shadow-mode form. Training
# uses whatever classes have >= MIN_CLASS_COUNT examples.
TARGET_DEVICES = [
    "PHOTOCELL", "HMD", "PROXIMITY", "LVDT", "ENCODER",
    "PRESSURE_SWITCH", "FLOW_SWITCH", "LASER",
]

CORS_ALLOW_ORIGINS = os.environ.get("PDM_CORS_ORIGINS", "").split(",") if os.environ.get("PDM_CORS_ORIGINS") else [
    "http://127.0.0.1:5500", "http://localhost:5500",  # local dev (Live Server)
]
# Only needed when a frontend on a different origin calls the API, e.g.
#   PDM_CORS_ORIGINS=http://10.0.4.12:8080,http://plant-pdm.local:8080
# Avoid "*" on a plant network.