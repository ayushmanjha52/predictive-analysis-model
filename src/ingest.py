"""
Data ingestion: reconciles Book1.xlsx (the primary, already-categorized
source) with the legacy master_events.csv (from earlier unification of
7 monthly files) -- WITHOUT double-counting.

CONFIRMED before writing this: 536 of Book1's 566 unique delay
descriptions are the exact same text as rows already in
legacy_master_events.csv -- these are two exports of the SAME
underlying data, not complementary datasets. Concatenating them
directly would have duplicated ~536 events.

Reconciliation approach:
  1. Book1.xlsx is treated as the primary source (cleaner, already has
     Category/Sub_Category, one consistent file).
  2. Only rows in legacy_master_events.csv whose reason_text does NOT
     already appear in Book1 are considered for inclusion.
  3. Of those, 3 were confirmed to be pure data artifacts (a stray Excel
     form header that leaked in during earlier unification -- "Name:
     Anshika Srivastava", "Designation: Assistant Manager", and a blank
     "Date" row) -- these are dropped entirely, not treated as delays.
  4. The remaining ~20 rows are genuine real events Book1 is missing
     (mostly April 2026, already correctly tagged with a field device)
     -- these are added in, normalized to the same schema as Book1.

Also normalizes, in one place, every messy-data issue found in Book1:
  - Category column mixes letter codes (F, FD, M, A, D, C, S) and full
    text ("Field Device", "Non-Field Device") in the SAME column.
  - Sub_Category has compound labels ("HMD / LVDT") and inconsistent
    device naming (PH vs Photocell, PX vs Proximity, LASER vs Laser).
  - Area has the same physical area written many different ways
    (bdm/BDM, furnace/Furnace/FURNACE/fce/Fce).
  - Date mixes string format ("15/11/2025") and real datetime objects
    in the same column.
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

# Category letter-code -> canonical text. Handles the mixed-encoding bug
# found directly in Book1: some rows use "F"/"FD", others use the full
# "Field Device" string, for the same meaning.
CATEGORY_MAP = {
    "F": "Field Device", "FD": "Field Device", "Field Device": "Field Device",
    "M": "Non-Field Device", "A": "Non-Field Device", "D": "Non-Field Device",
    "C": "Non-Field Device", "S": "Non-Field Device",
    "Non-Field Device": "Non-Field Device",
}

# Sub-category / device vocabulary normalization -- collapses both
# abbreviation variants (PH, PX, PS, FS) and casing variants (LASER vs
# Laser) to ONE canonical device name per concept.
DEVICE_NORMALIZE = {
    "PH": "PHOTOCELL", "PHOTOCELL": "PHOTOCELL",
    "PX": "PROXIMITY", "PROXIMITY": "PROXIMITY",
    "PS": "PRESSURE_SWITCH",
    "FS": "FLOW_SWITCH",
    "HMD": "HMD",
    "LVDT": "LVDT",
    "ENCODER": "ENCODER",
    "LASER": "LASER",
    "P": "PHOTOCELL",  # bare "P" appeared 36x in Sub_Category -- context
                        # (co-occurring with PH/Photocell counts) strongly
                        # suggests this is also photocell shorthand; flag
                        # for manual confirmation if this assumption is wrong
    "TT": "TT",
    "RFID": "RFID",
    "HIP": "HIP",
}

# Area name normalization -- collapses casing + a few known synonyms.
# Extend this map as new variants are found; unmatched values pass
# through title-cased rather than being dropped.
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
    """Collapses compound sub-categories (e.g. 'HMD / LVDT') to the
    first-listed device, then normalizes vocabulary via DEVICE_NORMALIZE.
    Non-device sub-categories (e.g. 'Automation / Sequence') pass through
    as None here -- they're not field devices, handled separately.

    FIX: lookup is case-insensitive now. Confirmed bug: raw Sub_Category
    values include title-case forms ('Photocell', 'Proximity') that
    didn't match the all-caps DEVICE_NORMALIZE keys, silently leaving
    29 real field-device rows with no resolved device."""
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


# Reason texts confirmed to be pure data artifacts (a stray Excel form
# header that leaked into the legacy CSV during earlier unification),
# not real delay events -- excluded unconditionally.
# Process outcomes/symptoms that were previously mistakenly treated as
# field devices in an earlier version of this project (confirmed: FMEA
# Severity/Occurrence/Detection ratings don't meaningfully apply to a
# symptom the way they apply to a component's failure mode). Never
# treated as a resolvable field device, even if a legacy source tagged
# them that way.
NON_DEVICE_CATEGORIES = {"SEQUENCE_BREAK", "OVERTRAVEL", "ROLLER_TABLE"}

KNOWN_ARTIFACT_REASONS = {
    "anshika srivastava",
    "assistant manager",
    "",
}


def load_legacy_supplementary_rows(book1_reasons_lower):
    """Only rows from the legacy master_events.csv whose reason_text is
    NOT already present in Book1 -- prevents double-counting the ~536
    rows confirmed to be duplicates. Artifact rows are dropped."""
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
        # FIX: legacy_master_events.csv's field_device column is ALREADY
        # canonical (e.g. "PRESSURE_SWITCH", "FLOW_SWITCH") -- it is the
        # OUTPUT of a previous normalization pass, not raw input. Running
        # it back through DEVICE_NORMALIZE (whose keys are raw
        # abbreviations like "PS"/"FS") silently dropped every row whose
        # legacy device name wasn't also a valid INPUT key. Confirmed:
        # 8 real Pressure_Switch/Flow_Switch rows lost this way. Fix: use
        # the legacy value directly (after stripping compound labels),
        # only falling back to DEVICE_NORMALIZE if it happens to still be
        # a raw abbreviation for some reason.
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
                f"{len(rows)} genuine supplementary rows added")
    return rows


def main():
    logger.info("Loading Book1.xlsx (primary source)...")
    book1_rows = load_book1()
    logger.info(f"Book1: {len(book1_rows)} rows")

    book1_reasons_lower = {r["reason_text"].strip().lower() for r in book1_rows}

    logger.info("Reconciling with legacy master_events.csv (supplementary rows only)...")
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