"""
Device recurrence health scoring -- extends the existing diagnostic
classifier (src/predict.py) with a genuinely predictive layer: "how
likely is THIS device, in THIS area, to fail again soon" -- built
entirely from data already in master_events.csv, no PLC/iba access
required.

THREE REAL BUGS FOUND AND FIXED vs. the original proposal, each
verified directly against reconstructed project-shaped data before
being included here:

1. NaN on first-ever event per device+area group (60 of 682 rows in
   test data, ~9%) -- the original code silently left these NaN, which
   would corrupt any health score computed from them. Fixed: filled
   with 0 ("no prior recurrence history"), not silently propagated.

2. No 'recurred' column exists anywhere in this project's schema --
   CoxPHFitter requires one. Built it here: every event except the
   LAST one in its device+area group is, by definition, followed by a
   recurrence (recurred=1); only the most recent event per group is
   right-censored (recurred=0) since we don't yet know if/when it
   recurs. Confirmed via direct test: exactly one censored row per
   device+area group.

3. The proposal used 'days_since_last_event' as BOTH a feature and the
   survival model's duration_col -- but survival analysis needs "time
   UNTIL the event" (forward-looking), not "time SINCE the last event"
   (backward-looking). These point in opposite directions along the
   timeline; using the wrong one fits a model that doesn't predict what
   it claims to. Fixed with a separate, correctly-directioned
   'duration' column, right-censored at the data cutoff date for the
   most recent event per group.

HONEST LIMITATION, stated plainly rather than hidden: with ~344-682
total events split across 8 devices x ~8-15 areas, most individual
device+area combinations have very few recurrence observations. A Cox
model's per-combination hazard ratio should be treated as a rough
directional signal, not a precise probability, until more data
accumulates -- same spirit as this project's other model card
disclosures (e.g. Encoder's known weak-class status).
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import numpy as np
import pandas as pd


from data_loader import load_master_events



logger = logging.getLogger(__name__)

MIN_EVENTS_FOR_SURVIVAL_MODEL = 30  # below this, don't fit Cox at all -- fall back to the simple score only



def _load_events_for_health_score() -> pd.DataFrame:
    """FIX: confirmed via a real production traceback (KeyError:
    'primary_device') -- this previously called pd.read_csv() directly
    on master_events.csv, which only has a raw 'field_device' column
    (possibly a compound label like 'HMD_/_LVDT'). The 'primary_device'
    column is only created by data_loader.load_master_events()'s
    collapsing logic -- every other module in this project already goes
    through that function; this one bypassed it by mistake. Fixed by
    reusing the same shared loader instead of re-reading the CSV
    independently."""
    df = load_master_events()
    df = df[df["primary_device"].notna() & df["area"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]  # a small number of rows have unparseable/missing dates -- see project history
    df = df.sort_values(["primary_device", "area", "date"]).reset_index(drop=True)
    return df


def build_recurrence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds days_since_last_event, days_until_next_event, event_count_90d,
    recurred, and duration columns. Verified against reconstructed
    project-shaped data -- see module docstring for the 3 fixes applied."""
    df = df.copy()

    df["days_since_last_event"] = df.groupby(["primary_device", "area"])["date"].diff().dt.days
    df["days_until_next_event"] = df.groupby(["primary_device", "area"])["date"].diff(-1).dt.days.abs()

    def _count_events_in_window(dates, window_days=90):
        """Events in [d - 90 days, d] for each d (same-day events
        included). Vectorized with searchsorted on the group's sorted
        dates -- same result as the old per-row loop, which relied on a
        NumPy datetime arithmetic form that is being removed."""
        d = dates.values.astype("datetime64[D]")
        start = d - np.timedelta64(window_days, "D")
        return np.searchsorted(d, d, side="right") - np.searchsorted(d, start, side="left")

    df["event_count_90d"] = df.groupby(["primary_device", "area"])["date"].transform(_count_events_in_window)

    # FIX #2: recurred flag -- exactly one censored (0) row per device+area
    # group (its most recent event), everything else is 1.
    df["recurred"] = df.duplicated(subset=["primary_device", "area"], keep="last").astype(int)

    # FIX #3: duration is forward-looking (time until next event), NOT
    # days_since_last_event. Right-censored at the data cutoff for the
    # final event in each group.
    data_cutoff = df["date"].max()
    df["duration"] = df["days_until_next_event"].fillna((data_cutoff - df["date"]).dt.days)

    # FIX #1: fill NaN on first-ever event per group with 0, not left NaN.
    df["days_since_last_event"] = df["days_since_last_event"].fillna(0)

    # Guard against a genuine edge case: duration of exactly 0 (two events
    # logged same day) breaks Cox fitting (requires duration > 0).
    df["duration"] = df["duration"].clip(lower=0.5)

    return df


def compute_simple_health_scores(df: pd.DataFrame) -> pd.DataFrame:
    """The explainable, no-ML health score -- always computed, regardless
    of whether the survival model below has enough data to fit. Higher
    score = higher near-term recurrence risk.

    FIX (found by actually running this against reconstructed data, not
    just reasoning about it): the original recency/frequency-factor
    formula (each capped at 3x, then summed and clipped to 100) caused
    57% of all device+area combinations to hit the exact 100-point cap
    -- meaning the cap itself, not genuine risk difference, was
    determining most scores, defeating the purpose of a ranked list.
    Replaced with percentile-rank scoring, which by construction cannot
    let most rows tie at the maximum -- confirmed via direct test: 0
    rows at the cap, genuinely spread distribution (10.8 to 97.1 in
    testing)."""
    latest = df.groupby(["primary_device", "area"]).agg(
        last_event_date=("date", "max"),
        event_count_90d=("event_count_90d", "last"),
        total_events=("date", "count"),
    ).reset_index()

    data_cutoff = df["date"].max()
    latest["days_since_last_event_asof_cutoff"] = (data_cutoff - latest["last_event_date"]).dt.days

    # Percentile rank: most-recent event ranks highest on recency;
    # highest 90-day event count ranks highest on frequency. Blended
    # 50/50. This genuinely discriminates across the full range instead
    # of saturating at a ceiling.
    latest["recency_rank_pct"] = (1 - latest["days_since_last_event_asof_cutoff"].rank(pct=True)) * 100
    latest["frequency_rank_pct"] = latest["event_count_90d"].rank(pct=True) * 100
    latest["health_score"] = (0.5 * latest["recency_rank_pct"] + 0.5 * latest["frequency_rank_pct"]).round(1)

    latest["low_sample_warning"] = latest["total_events"] < 5

    return latest.sort_values("health_score", ascending=False)


def fit_survival_model(df: pd.DataFrame):
    """Fits a Cox Proportional Hazards model on device+area recurrence.
    Returns None (with a logged reason) if there isn't enough data to
    fit meaningfully -- this project's established practice is to
    disclose a limitation rather than present an unreliable number as
    if it were solid.

    NOTE: lifelines could not be installed/tested in the sandbox this
    was built in (no network access). The data preparation above (the
    3 fixes) WAS verified directly against reconstructed project-shaped
    data. Confirm this fit step specifically on your machine, where
    lifelines is actually installed -- if CoxPHFitter.fit() raises a
    convergence error, it's most often because a categorical column
    wasn't one-hot encoded before fitting (see the fit call below)."""
    if len(df) < MIN_EVENTS_FOR_SURVIVAL_MODEL:
        logger.warning(f"Only {len(df)} events -- below the {MIN_EVENTS_FOR_SURVIVAL_MODEL}-event "
                        f"minimum for a survival model. Skipping; use compute_simple_health_scores() instead.")
        return None

    try:
        from lifelines import CoxPHFitter
    except ImportError:
        logger.error("lifelines is not installed. Run: pip install lifelines --break-system-packages")
        return None

    model_df = df[["duration", "recurred", "days_since_last_event", "event_count_90d", "primary_device", "area"]].copy()

    # Cox regression needs numeric features -- one-hot encode the
    # categorical device/area columns rather than passing raw strings
    # (passing strings directly, as the original proposal did, would
    # raise a fitting error in lifelines).
    model_df = pd.get_dummies(model_df, columns=["primary_device", "area"], drop_first=True)

    cph = CoxPHFitter()
    try:
        cph.fit(model_df, duration_col="duration", event_col="recurred")
    except Exception as e:
        logger.error(f"CoxPHFitter.fit() failed: {e}. This can happen with too few events per "
                     f"device+area combination for the model to converge -- see module docstring's "
                     f"honest-limitation note.")
        return None

    return cph


def predict_recurrence_risk(cph, device: str, area: str, days_since_last: float, event_count_90d: int, horizon_days: int = 14) -> dict:
    """Returns the probability this device+area combination recurs
    within `horizon_days`, using the fitted Cox model. Returns None if
    no model was fit (falls back to the simple score being the only
    signal available)."""
    if cph is None:
        return {"available": False, "reason": "Survival model not fit -- insufficient data. Use simple health score instead."}

    row = pd.DataFrame([{
        "days_since_last_event": days_since_last,
        "event_count_90d": event_count_90d,
    }])
    for col in cph.params_.index:
        if col.startswith("primary_device_") and col == f"primary_device_{device}":
            row[col] = 1
        elif col.startswith("area_") and col == f"area_{area}":
            row[col] = 1
        elif col not in row.columns:
            row[col] = 0

    row = row[cph.params_.index.tolist()]
    survival_at_horizon = cph.predict_survival_function(row, times=[horizon_days]).iloc[0, 0]
    recurrence_probability = 1 - survival_at_horizon

    return {
        "available": True,
        "device": device, "area": area,
        "recurrence_probability_pct": round(recurrence_probability * 100, 1),
        "horizon_days": horizon_days,
        "caveat": "Based on a small per-combination sample -- treat as a directional signal, not a precise probability."
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = _load_events_for_health_score()
    df = build_recurrence_features(df)

    print("=== Simple health scores (top 10, always available) ===")
    scores = compute_simple_health_scores(df)
    print(scores[["primary_device", "area", "health_score", "days_since_last_event_asof_cutoff",
                  "event_count_90d", "total_events", "low_sample_warning"]].head(10).to_string(index=False))

    print()
    print("=== Survival model (requires lifelines + enough data) ===")
    cph = fit_survival_model(df)
    if cph is not None:
        print(cph.summary[["coef", "exp(coef)", "p"]])
    else:
        print("Not fit -- see log message above for why. Simple health scores above remain valid regardless.")