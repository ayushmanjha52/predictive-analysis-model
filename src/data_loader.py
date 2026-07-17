"""Load the unified master_events.csv. Single source of truth for how
the rest of the project reads event data -- every module imports
load_master_events() from here rather than calling pd.read_csv directly,
so a schema fix here propagates everywhere at once.
"""
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import pandas as pd
from config import MASTER_EVENTS_PATH

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "month", "mins", "reason_text", "field_device",
                    "area", "tag_source", "source_file"}


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

    logger.info(f"Loaded {len(df)} events from {MASTER_EVENTS_PATH.name}")
    return df