"""
Per-device monthly volume forecasting -- answers "which devices will
have the most delay in the next 3-4 months" per your requirement.

Uses month LABEL column (never re-derived from a date column -- this
project previously found an entire month's date column silently NaN,
which would corrupt any date-based grouping). Reports 3-month rolling
average as the primary number, with a linear trend only when it
actually meets a reliability bar -- never presented with false
confidence.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
from typing import Optional
import pandas as pd
from sklearn.linear_model import LinearRegression

import config
from data_loader import load_master_events

logger = logging.getLogger(__name__)

R2_TRUST_THRESHOLD = 0.5
MIN_MONTHS_FOR_TREND = 6


def _monthly_series(df, device=None):
    d = df.copy()
    if device:
        d = d[d["primary_device"] == device.upper()]
    else:
        d = d[d["primary_device"].notna()]
    d["mins"] = pd.to_numeric(d["mins"], errors="coerce")
    monthly = d.groupby("month")["mins"].sum().reset_index(name="mins")
    monthly["month"] = pd.Categorical(monthly["month"], categories=config.MONTH_ORDER, ordered=True)
    monthly = monthly.sort_values("month").dropna(subset=["month"]).reset_index(drop=True)
    return monthly


def forecast_next_month(df=None, device: Optional[str] = None, forecast_months: int = 4) -> dict:
    if df is None:
        df = load_master_events()

    monthly = _monthly_series(df, device=device)
    if len(monthly) == 0:
        return {"device": device, "message": "No data available for this device."}
    if len(monthly) < 3:
        return {
            "device": device,
            "message": "Not enough months for reliable forecasting (need at least 3).",
            "monthly_totals": monthly.to_dict(orient="records"),
        }

    monthly["time_index"] = range(len(monthly))
    X = monthly[["time_index"]]
    y = monthly["mins"]

    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    future_index = pd.DataFrame({"time_index": range(len(monthly), len(monthly) + forecast_months)})
    trend_forecast_series = model.predict(future_index)
    trend_forecast_total = float(sum(max(0, v) for v in trend_forecast_series))

    rolling_avg = monthly["mins"].tail(3).mean()
    rolling_avg_forecast_total = float(rolling_avg * forecast_months)

    trend_reliable = (r2 >= R2_TRUST_THRESHOLD) and (len(monthly) >= MIN_MONTHS_FOR_TREND)

    return {
        "device": device or "ALL_FIELD_DEVICES",
        "monthly_totals": monthly[["month", "mins"]].to_dict(orient="records"),
        "rolling_avg_per_month": round(rolling_avg, 1),
        "rolling_avg_forecast_next_n_months": round(rolling_avg_forecast_total, 1),
        "trend_forecast_next_n_months": round(trend_forecast_total, 1),
        "trend_r_squared": round(r2, 3),
        "trend_reliable": trend_reliable,
        "forecast_months": forecast_months,
        "recommended_forecast": round(rolling_avg_forecast_total, 1) if not trend_reliable else round(trend_forecast_total, 1),
        "note": (
            f"Trend R^2={r2:.2f} on {len(monthly)} months -- "
            + ("below reliability threshold, use rolling_avg_forecast instead."
               if not trend_reliable else "meets reliability bar for this sample size.")
        ),
    }


def forecast_all_devices(df=None, forecast_months: int = 4) -> dict:
    if df is None:
        df = load_master_events()
    devices = sorted(df["primary_device"].dropna().unique().tolist())
    results = {d: forecast_next_month(df, device=d, forecast_months=forecast_months) for d in devices}

    # Rank by recommended forecast -- directly answers "which device
    # will have the most delay in the next N months."
    ranked = sorted(
        [(d, r.get("recommended_forecast", 0)) for d, r in results.items() if "recommended_forecast" in r],
        key=lambda x: -x[1]
    )
    return {
        "forecasts": results,
        "ranked_by_expected_delay": [{"device": d, "expected_minutes": v} for d, v in ranked],
        "forecast_months": forecast_months,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(forecast_all_devices(), indent=2, default=str))