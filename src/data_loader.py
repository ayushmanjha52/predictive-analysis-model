"""Load event data. Single source of truth for how the rest of the
project reads event data -- every module imports from here rather than
calling pd.read_csv directly, so a schema fix here propagates
everywhere at once.

Two kinds of events, one schema:
  - HOME site (Combi Mill): data/master_events.csv -- hand-tagged,
    has delay minutes, is the honest accuracy benchmark, and is what
    /log_event appends to.
  - EXTERNAL fleet (public power-plant data, see external_data.py):
    data/external/external_events.csv.gz -- weakly labeled, no delay
    minutes, but carries objective severity signals (reactor_trip,
    power_lost_pct) and site/plant metadata.
Every row from either source has `site` and `plant_type` columns, so
any consumer can filter or group across sites.
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import pandas as pd
import config
from config import MASTER_EVENTS_PATH

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "month", "mins", "reason_text", "field_device",
                    "area", "tag_source", "source_file"}

SITE_COLUMNS = ["site", "plant_type", "state", "country", "reactor_trip",
                "grid_related", "power_lost_pct"]


def primary_device(val):
    """Collapses any compound label (e.g. 'HMD_/_LVDT') to the
    first-listed device. ingest.py already does this at write time, but
    this is kept here too as a defensive second layer -- if any future
    data source ever bypasses ingest.py and writes directly to
    master_events.csv, this still protects every downstream consumer."""
    if pd.isna(val) or not val:
        return None
    return str(val).split("_/_")[0].split("/")[0].strip().upper()


def load_master_events() -> pd.DataFrame:
    """The home site's (Combi Mill) own event log."""
    if not MASTER_EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"{MASTER_EVENTS_PATH} not found. Run `python src/ingest.py` first "
            f"to build it from data/raw_delays.xlsx."
        )
    df = pd.read_csv(MASTER_EVENTS_PATH)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"master_events.csv is missing required columns: {missing}. "
            f"Re-run ingest.py -- the schema may have changed."
        )

    df["primary_device"] = df["field_device"].apply(primary_device)
    df["mins"] = pd.to_numeric(df["mins"], errors="coerce")
    df["site"] = config.HOME_SITE
    df["plant_type"] = config.HOME_PLANT_TYPE
    df["country"] = "IND"
    for col in ("state", "reactor_trip", "grid_related", "power_lost_pct"):
        if col not in df.columns:
            df[col] = None

    logger.info(f"Loaded {len(df)} events from {MASTER_EVENTS_PATH.name}")
    return df


def load_external_events() -> pd.DataFrame:
    """Public power-plant fleet events. Returns an empty frame (same
    columns) if external_data.py hasn't been run -- the app still works
    on home data alone."""
    if not config.EXTERNAL_EVENTS_PATH.exists():
        logger.warning(f"{config.EXTERNAL_EVENTS_PATH} not found -- run `python src/external_data.py` "
                       f"to add public power-plant data. Continuing with home-site data only.")
        return pd.DataFrame(columns=sorted(REQUIRED_COLUMNS | set(SITE_COLUMNS) | {"primary_device"}))
    df = pd.read_csv(config.EXTERNAL_EVENTS_PATH, low_memory=False)
    df["primary_device"] = df["field_device"].apply(primary_device)
    df["mins"] = pd.to_numeric(df["mins"], errors="coerce")
    for col in ("reactor_trip", "grid_related"):
        df[col] = df[col].astype(str).str.lower() == "true"
    logger.info(f"Loaded {len(df)} external events from {config.EXTERNAL_EVENTS_PATH.name}")
    return df


def load_all_events() -> pd.DataFrame:
    """Home + external fleet, one frame."""
    home = load_master_events()
    ext = load_external_events()
    if len(ext) == 0:
        return home
    return pd.concat([home, ext], ignore_index=True, sort=False)
