"""
Builds data/master_events.csv from the raw mill delay files.

Sources:
  1. raw_delays.xlsx (primary; already categorized).
  2. legacy_master_events.csv: only rows whose reason text is NOT in the
     primary file are added (the two files are ~95% overlapping exports
     of the same delays). Known form-header artifacts are dropped.

Normalization:
  - Category mixes letter codes (F, FD, M, A, D, C, S) and full text.
  - Sub_Category has compound labels ("HMD / LVDT") and abbreviations
    (PH, PX, PS, FS) -> one canonical device name each.
  - Area spellings (bdm/BDM, furnace/fce, ...) -> one name each.
  - Dates mix strings ("15/11/2025") and datetime objects.
"""
import logging
import datetime as dt
from pathlib import Path

import pandas as pd
import openpyxl

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
RAW_DELAYS_PATH = DATA_DIR / "raw_delays.xlsx"
LEGACY_MASTER_PATH = DATA_DIR / "legacy_master_events.csv"
OUTPUT_PATH = DATA_DIR / "master_events.csv"

# Category letter codes -> canonical text.
CATEGORY_MAP = {
    "F": "Field Device", "FD": "Field Device", "Field Device": "Field Device",
    "M": "Non-Field Device", "A": "Non-Field Device", "D": "Non-Field Device",
    "C": "Non-Field Device", "S": "Non-Field Device",
    "Non-Field Device": "Non-Field Device",
}

# Sub-category abbreviations / casing -> canonical device name.
DEVICE_NORMALIZE = {
    "PH": "PHOTOCELL", "PHOTOCELL": "PHOTOCELL",
    "PX": "PROXIMITY", "PROXIMITY": "PROXIMITY",
    "PS": "PRESSURE_SWITCH",
    "FS": "FLOW_SWITCH",
    "HMD": "HMD",
    "LVDT": "LVDT",
    "ENCODER": "ENCODER",
    "LASER": "LASER",
    "P": "PHOTOCELL",  # bare "P" is photocell shorthand in Sub_Category
    "TT": "TT",
    "RFID": "RFID",
    "HIP": "HIP",
}

# Area synonyms; unmatched values pass through upper-cased.
AREA_SYNONYMS = {
    "fce": "FURNACE", "furnace": "FURNACE",
    "bdm": "BDM",
    "cooling bed": "COOLING_BED", "cb": "COOLING_BED",
    "kocks": "KOCKS",
    "crane": "CRANE",
    "elec": "ELECTRICAL",
    "mill": "MILL",
    "saw": "SAW", "saws": "SAW",
    "shear": "SHEAR", "shears": "SHEAR",
    "bundling": "BUNDLING", "bundling area": "BUNDLING",
}


def normalize_category(val):
    if pd.isna(val):
        return None
    return CATEGORY_MAP.get(str(val).strip(), str(val).strip())


def primary_device(val):
    """First device of a compound sub-category, normalized via
    DEVICE_NORMALIZE (case-insensitive). Non-device sub-categories -> None."""
    if pd.isna(val):
        return None
    first = str(val).split("/")[0].strip()
    return DEVICE_NORMALIZE.get(first.upper(), None)


def normalize_area(val):
    if pd.isna(val):
        return None
    key = str(val).strip().lower()
    return AREA_SYNONYMS.get(key, str(val).strip().upper())


def normalize_date(val):
    if isinstance(val, dt.datetime):
        return val.date()
    if isinstance(val, str):
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
            try:
                return dt.datetime.strptime(val.strip(), fmt).date()
            except ValueError:
                continue
    return None


def month_label(d):
    if d is None:
        return None
    return d.strftime("%b-%y")


def load_book1():
    wb = openpyxl.load_workbook(RAW_DELAYS_PATH, data_only=True)
    ws = wb["Sheet1"]
    rows = []
    for r in ws.iter_rows(min_row=2, values_only=True):
        date_raw, desc, area, mins, category, subcat = (list(r) + [None] * 6)[:6]
        if not desc:
            continue
        d = normalize_date(date_raw)
        rows.append({
            "date": d,
            "month": month_label(d),
            "mins": mins,
            "reason_text": str(desc).strip(),
            "area": normalize_area(area),
            "category": normalize_category(category),
            "field_device": primary_device(subcat),
            "tag_source": "book1_categorized",
            "source_file": "Book1.xlsx",
        })
    return rows


# Process symptoms that are not field devices; never used as a device label.
NON_DEVICE_CATEGORIES = {"SEQUENCE_BREAK", "OVERTRAVEL", "ROLLER_TABLE"}

# Form-header rows that leaked into the legacy export.
KNOWN_ARTIFACT_REASONS = {
    "anshika srivastava",
    "assistant manager",
    "",
}


def load_legacy_supplementary_rows(book1_reasons_lower):
    """Legacy rows whose reason text is not already in the primary file."""
    legacy = pd.read_csv(LEGACY_MASTER_PATH)
    rows = []
    n_skipped_duplicate = 0
    n_skipped_artifact = 0
    for _, row in legacy.iterrows():
        reason = str(row.get("reason_text", "")).strip()
        reason_lower = reason.lower()
        if reason_lower in KNOWN_ARTIFACT_REASONS:
            n_skipped_artifact += 1
            continue
        if reason_lower in book1_reasons_lower:
            n_skipped_duplicate += 1
            continue
        d = normalize_date(row.get("date"))
        # Legacy field_device values are already canonical; normalize only
        # if a raw abbreviation slipped through.
        legacy_device_raw = row.get("field_device")
        legacy_device = None
        if pd.notna(legacy_device_raw):
            first = str(legacy_device_raw).split("_/_")[0].strip().upper()
            if first not in NON_DEVICE_CATEGORIES:
                legacy_device = DEVICE_NORMALIZE.get(first, first)  # pass through if already canonical
        rows.append({
            "date": d,
            "month": row.get("month") or month_label(d),
            "mins": row.get("mins"),
            "reason_text": reason,
            "area": None,
            "category": "Field Device" if legacy_device else "Non-Field Device",
            "field_device": legacy_device,
            "tag_source": "legacy_supplementary",
            "source_file": "legacy_master_events.csv",
        })

    logger.info(f"Legacy reconciliation: {n_skipped_duplicate} rows skipped as duplicates of Book1, "
                f"{n_skipped_artifact} rows skipped as known artifacts, "
                f"{len(rows)} supplementary rows added")
    return rows


def main():
    logger.info("Loading Book1.xlsx (primary source)...")
    book1_rows = load_book1()
    logger.info(f"Book1: {len(book1_rows)} rows")

    book1_reasons_lower = {r["reason_text"].strip().lower() for r in book1_rows}

    logger.info("Adding legacy rows not present in the primary file...")
    supplementary_rows = load_legacy_supplementary_rows(book1_reasons_lower)

    all_rows = book1_rows + supplementary_rows
    df = pd.DataFrame(all_rows)

    logger.info(f"Final reconciled dataset: {len(df)} rows "
                f"({len(book1_rows)} from Book1 + {len(supplementary_rows)} supplementary)")

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    main()