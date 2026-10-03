"""
Device recurrence health scoring: how likely a device, in a given area,
is to fail again soon. Built from master_events.csv only.

Two outputs:
  - a simple, explainable health score (always available), and
  - a Cox proportional-hazards recurrence model (needs `lifelines` and
    at least MIN_EVENTS_FOR_SURVIVAL_MODEL events).

Survival framing per device+area group:
  - recurred = 1 for every event followed by another event in the same
    group; the most recent event is right-censored (recurred = 0).
  - duration = days UNTIL the next event (forward-looking), censored at
    the data cutoff for the most recent event.

Limitation: most device+area combinations have only a handful of
events, so per-combination hazard estimates are directional signals,
not precise probabilities.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import numpy as np
import pandas as pd

from data_loader import load_master_events

logger = logging.getLogger(__name__)

MIN_EVENTS_FOR_SURVIVAL_MODEL = 30  # below this, only the simple score is used


def _load_events_for_health_score() -> pd.DataFrame:
    df = load_master_events()
    df = df[df["primary_device"].notna() & df["area"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    df = df.sort_values(["primary_device", "area", "date"]).reset_index(drop=True)
    return df


def build_recurrence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds days_since_last_event, days_until_next_event, event_count_90d,
    recurred and duration columns. Expects df sorted by device, area, date."""
    df = df.copy()
    groups = df.groupby(["primary_device", "area"])

    df["days_since_last_event"] = groups["date"].diff().dt.days
    df["days_until_next_event"] = groups["date"].diff(-1).dt.days.abs()

    def _count_events_in_window(dates, window_days=90):
        """Events in [d - 90 days, d] for each d, same-day events included."""
        d = dates.values.astype("datetime64[D]")
        start = d - np.timedelta64(window_days, "D")
        return np.searchsorted(d, d, side="right") - np.searchsorted(d, start, side="left")

    df["event_count_90d"] = groups["date"].transform(_count_events_in_window)

    # Most recent event per group is censored; all others recurred.
    df["recurred"] = df.duplicated(subset=["primary_device", "area"], keep="last").astype(int)

    data_cutoff = df["date"].max()
    df["duration"] = df["days_until_next_event"].fillna((data_cutoff - df["date"]).dt.days)

    # First event in a group has no previous event.
    df["days_since_last_event"] = df["days_since_last_event"].fillna(0)

    # Cox fitting requires duration > 0 (two events on the same day give 0).
    df["duration"] = df["duration"].clip(lower=0.5)

    return df


def compute_simple_health_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Explainable health score per device+area; higher = higher near-term
    recurrence risk. 50/50 blend of recency and 90-day frequency, each as
    a percentile rank so scores spread across the range instead of
    saturating at a cap."""
    latest = df.groupby(["primary_device", "area"]).agg(
        last_event_date=("date", "max"),
        event_count_90d=("event_count_90d", "last"),
        total_events=("date", "count"),
    ).reset_index()

    data_cutoff = df["date"].max()
    latest["days_since_last_event_asof_cutoff"] = (data_cutoff - latest["last_event_date"]).dt.days

    latest["recency_rank_pct"] = (1 - latest["days_since_last_event_asof_cutoff"].rank(pct=True)) * 100
    latest["frequency_rank_pct"] = latest["event_count_90d"].rank(pct=True) * 100
    latest["health_score"] = (0.5 * latest["recency_rank_pct"] + 0.5 * latest["frequency_rank_pct"]).round(1)

    latest["low_sample_warning"] = latest["total_events"] < 5

    return latest.sort_values("health_score", ascending=False)


def fit_survival_model(df: pd.DataFrame):
    """Fits a Cox proportional-hazards model on device+area recurrence.
    Returns None (and logs why) if there isn't enough data or lifelines
    isn't installed."""
    if len(df) < MIN_EVENTS_FOR_SURVIVAL_MODEL:
        logger.warning(f"Only {len(df)} events -- below the {MIN_EVENTS_FOR_SURVIVAL_MODEL}-event "
                        f"minimum for a survival model. Skipping; use compute_simple_health_scores() instead.")
        return None

    try:
        from lifelines import CoxPHFitter
    except ImportError:
        logger.error("lifelines is not installed. Run: pip install lifelines")
        return None

    model_df = df[["duration", "recurred", "days_since_last_event", "event_count_90d", "primary_device", "area"]].copy()
    model_df = pd.get_dummies(model_df, columns=["primary_device", "area"], drop_first=True)

    cph = CoxPHFitter()
    try:
        cph.fit(model_df, duration_col="duration", event_col="recurred")
    except Exception as e:
        logger.error(f"CoxPHFitter.fit() failed: {e}. This usually means too few events per "
                     f"device+area combination for the model to converge.")
        return None

    return cph


def predict_recurrence_risk(cph, device: str, area: str, days_since_last: float, event_count_90d: int, horizon_days: int = 14) -> dict:
    """Probability this device+area combination recurs within `horizon_days`."""
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
    df = build_recurrence_features(_load_events_for_health_score())

    print("=== Simple health scores (top 10) ===")
    scores = compute_simple_health_scores(df)
    print(scores[["primary_device", "area", "health_score", "days_since_last_event_asof_cutoff",
                  "event_count_90d", "total_events", "low_sample_warning"]].head(10).to_string(index=False))

    print("\n=== Survival model ===")
    cph = fit_survival_model(df)
    print(cph.summary[["coef", "exp(coef)", "p"]] if cph is not None else "Not fit (see log above).")
