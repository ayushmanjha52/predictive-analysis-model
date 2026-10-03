"""
FastAPI backend -- Field-Device Predictive Maintenance: Combi Mill +
power-generation fleet insights.

Home site (Combi Mill) endpoints:
  GET  /health                - liveness + model-loaded check
  GET  /model_info            - manifest: winning model, backtest accuracy, classes
  POST /predict               - classify delay text (+ optional area, plant_type), with expected duration
  POST /log_event             - classify AND record as a real, timestamped event
  GET  /stats                 - area-aware Pareto + FMEA fusion + recommendations
  GET  /forecast              - per-device volume forecast
  GET  /forecast/all_devices  - every device, ranked
  GET  /events                - paginated/filterable event log (site=COMBI_MILL | ALL | <plant name>)
  GET  /device_area_breakdown - device x area cross-tab
  GET  /area_risk, /unresolved_hotspots, /review_queue, /health_score
  POST /shadow_predict, /shadow_resolve; GET /shadow_stats

Fleet-wide (public power-plant data, see src/external_data.py):
  GET  /sites                  - every site in the dataset, by plant type
  GET  /fleet/overview         - headline numbers across all US power reactors
  GET  /fleet/device_ranking   - device classes ranked across the fleet vs. Combi Mill
  GET  /fleet/failure_modes    - what actually went wrong, mined from narratives
  GET  /fleet/lessons          - cross-site lessons for the Combi Mill's devices
  GET  /fleet/trend            - yearly fleet trend
  GET  /fleet/grid             - grid / offsite-power event insights
  GET  /fleet/plant_reliability- device events per GW, per plant
  GET  /power_plants           - global installed base (WRI database), optional ?country=IND

The dashboard (frontend/) is served from the same app at "/".

Run: uvicorn app:app --reload --port 8000
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import json
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import fleet_insights
from data_loader import load_master_events, load_all_events
from fmea_risk import load_fmea_scores, fuse_pareto_with_fmea, get_recommendation
from forecasting import forecast_next_month, forecast_all_devices
from predict import get_predictor
from health_score import (_load_events_for_health_score, build_recurrence_features,
                          compute_simple_health_scores, fit_survival_model,
                          predict_recurrence_risk, MIN_EVENTS_FOR_SURVIVAL_MODEL)

FRONTEND_DIR = Path(__file__).parent / "frontend"

app = FastAPI(
    title="Field-Device Predictive Maintenance API -- Combi Mill + Power Fleet",
    version="3.0.0",
    description="Classifies delay causes (device + area), forecasts delay volume, fuses empirical "
                "data with FMEA risk, and benchmarks against public power-plant fleet data "
                "to support a 50% field-device delay reduction target.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_events_cache = None
_predictor = None


def get_events():
    global _events_cache
    if _events_cache is None:
        _events_cache = load_master_events()
    return _events_cache


def get_predictor_safe():
    global _predictor
    if _predictor is None:
        try:
            _predictor = get_predictor()
        except FileNotFoundError:
            raise HTTPException(status_code=503,
                                 detail="Model not trained yet. Run `python src/train.py` first.")
    return _predictor


# ---------------------------------------------------------------- Schemas --
class PredictRequest(BaseModel):
    reason_text: str = Field(..., min_length=3, max_length=2000)
    area: Optional[str] = Field(None, description="e.g. BDM, SAW, COOLING_BED")
    mins: Optional[float] = Field(None, ge=0)
    plant_type: Optional[str] = Field(None, description="Defaults to the Combi Mill (STEEL_ROLLING_MILL). "
                                                         "e.g. NUCLEAR_POWER for a power-plant delay.")


# --------------------------------------------------------------- Endpoints --
@app.get("/api")
def api_root():
    return {"message": "Field-Device Predictive Maintenance API", "docs": "/docs", "health": "/health",
            "dashboard": "/"}


@app.get("/area_risk")
def area_risk(top_n: int = Query(5, ge=1, le=50)):
    """Areas ranked by total field-device delay minutes."""
    df = get_events()
    tagged = df[df["primary_device"].notna() & df["area"].notna()]
    area_totals = tagged.groupby("area").agg(
        total_minutes=("mins", "sum"), events=("mins", "count"),
    ).reset_index().sort_values("total_minutes", ascending=False).head(top_n)
    total_all = area_totals["total_minutes"].sum()
    area_totals["pct_share"] = (area_totals["total_minutes"] / total_all * 100).round(1) if total_all else 0
    return {"top_risk_areas": area_totals.to_dict(orient="records")}


@app.get("/unresolved_hotspots")
def unresolved_hotspots(top_n: int = Query(5, ge=1, le=20)):
    """Areas ranked by delay minutes not yet attributed to a field device."""
    df = get_events()
    unresolved = df[df["primary_device"].isna() & df["area"].notna()]
    hotspots = unresolved.groupby("area").agg(
        total_minutes=("mins", "sum"), events=("mins", "count"),
    ).reset_index().sort_values("total_minutes", ascending=False).head(top_n)
    return {"top_unresolved_hotspots": hotspots.to_dict(orient="records")}


@app.get("/review_queue")
def review_queue(limit: int = Query(50, ge=1, le=500)):
    """Unresolved delays ranked by minutes, for manual tagging."""
    df = get_events()
    unresolved = df[df["primary_device"].isna()].copy()
    unresolved = unresolved.sort_values("mins", ascending=False).head(limit)
    records = unresolved[["date", "month", "mins", "reason_text", "area"]].fillna(
        {"area": "UNKNOWN", "date": "", "month": "", "reason_text": "(no description)"}
    )
    records = records.astype(object).where(records.notna(), None).to_dict(orient="records")
    return {
        "total_unresolved": int(df["primary_device"].isna().sum()),
        "total_unresolved_minutes": float(df[df["primary_device"].isna()]["mins"].sum()),
        "queue": records,
    }


@app.get("/health")
def health():
    model_ready = config.MODEL_PATH.exists() and config.VECTORIZER_PATH.exists()
    return {"status": "ok", "model_ready": model_ready,
            "external_data": config.EXTERNAL_EVENTS_PATH.exists()}


@app.get("/model_info")
def model_info():
    if not config.MANIFEST_PATH.exists():
        raise HTTPException(status_code=503, detail="No training manifest found. Run training first.")
    with open(config.MANIFEST_PATH) as f:
        return json.load(f)


@app.post("/predict")
def predict(req: PredictRequest):
    predictor = get_predictor_safe()
    try:
        result = predictor.predict(req.reason_text, area=req.area, plant_type=req.plant_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    result["recommendation"] = get_recommendation(result["predicted_device"])
    return result


@app.post("/log_event")
def log_event(req: PredictRequest):
    """Classify and append to master_events.csv. These rows are tagged
    tag_source='live_prediction' and excluded from training."""
    predictor = get_predictor_safe()
    now = datetime.now()
    result = predictor.predict(req.reason_text, area=req.area)

    new_row = {
        "date": now.strftime("%Y-%m-%d"),
        "month": now.strftime("%b-%y"),
        "mins": req.mins if req.mins is not None else "",
        "reason_text": req.reason_text,
        "field_device": result["predicted_device"],
        "area": req.area or "",
        "tag_source": "live_prediction",
        "source_file": "live_log_event",
    }

    df = pd.read_csv(config.MASTER_EVENTS_PATH)
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(config.MASTER_EVENTS_PATH, index=False)

    global _events_cache
    _events_cache = None

    result["recommendation"] = get_recommendation(result["predicted_device"])
    result["logged"] = True
    result["logged_at"] = now.isoformat()
    result["logged_at_display"] = now.strftime("%Y-%m-%d %H:%M")
    return result


# ------------------------------------------------------ Shadow-mode --
# The prediction is logged first without being shown; the reviewer's
# independent diagnosis is recorded separately, then compared.

class ShadowPredictRequest(BaseModel):
    reason_text: str = Field(..., min_length=3, max_length=2000)
    area: Optional[str] = None
    mins: Optional[float] = Field(None, ge=0)


class ShadowResolveRequest(BaseModel):
    shadow_id: str = Field(..., description="The id returned by /shadow_predict")
    actual_device: str = Field(..., description="What the reviewer actually found -- their independent diagnosis")
    reviewer_note: Optional[str] = Field(None, max_length=1000)


SHADOW_COLUMNS = ["shadow_id", "logged_at", "reason_text", "area", "mins",
                  "predicted_device", "confidence", "actual_device",
                  "resolved_at", "reviewer_note", "agreement"]


def _load_shadow_log() -> pd.DataFrame:
    if not config.SHADOW_LOG_PATH.exists():
        return pd.DataFrame(columns=SHADOW_COLUMNS)
    return pd.read_csv(config.SHADOW_LOG_PATH, dtype={"shadow_id": str})


def _is_resolved(log: pd.DataFrame) -> pd.Series:
    """Empty CSV cells read back as NaN, so check for a real value."""
    return log["resolved_at"].notna() & (log["resolved_at"].astype(str).str.strip() != "")


@app.post("/shadow_predict")
def shadow_predict(req: ShadowPredictRequest):
    """Step 1: record the model's prediction without revealing it."""
    predictor = get_predictor_safe()
    now = datetime.now()
    result = predictor.predict(req.reason_text, area=req.area)
    shadow_id = str(uuid.uuid4())[:8]

    new_row = {
        "shadow_id": shadow_id,
        "logged_at": now.isoformat(),
        "reason_text": req.reason_text,
        "area": req.area or "",
        "mins": req.mins if req.mins is not None else "",
        "predicted_device": result["predicted_device"],
        "confidence": result["confidence"],
        "actual_device": "", "resolved_at": "", "reviewer_note": "", "agreement": "",
    }
    log = _load_shadow_log()
    log = pd.concat([log, pd.DataFrame([new_row])], ignore_index=True)
    log.to_csv(config.SHADOW_LOG_PATH, index=False)

    return {"shadow_id": shadow_id, "logged": True,
            "message": "Prediction recorded. Determine the actual cause independently, then submit via /shadow_resolve."}


@app.post("/shadow_resolve")
def shadow_resolve(req: ShadowResolveRequest):
    """Step 2: record the reviewer's independent actual diagnosis and
    compute whether it agreed with the model's (now-revealed) prediction."""
    log = _load_shadow_log()
    match = log["shadow_id"].astype(str) == req.shadow_id
    if not match.any():
        raise HTTPException(status_code=404, detail=f"No shadow entry found for id '{req.shadow_id}'.")

    now = datetime.now()
    idx = log[match].index[0]
    predicted = str(log.loc[idx, "predicted_device"]).upper()
    actual = req.actual_device.upper().strip()
    agreement = predicted == actual

    # All-empty columns are read as float64; cast before writing strings.
    for col in ("actual_device", "resolved_at", "reviewer_note", "agreement"):
        log[col] = log[col].astype(object)

    log.loc[idx, "actual_device"] = actual
    log.loc[idx, "resolved_at"] = now.isoformat()
    log.loc[idx, "reviewer_note"] = req.reviewer_note or ""
    log.loc[idx, "agreement"] = agreement
    log.to_csv(config.SHADOW_LOG_PATH, index=False)

    return {
        "shadow_id": req.shadow_id,
        "predicted_device": predicted,
        "actual_device": actual,
        "agreement": bool(agreement),
        "resolved": True,
    }


@app.get("/shadow_stats")
def shadow_stats():
    """Agreement rate over resolved shadow-mode entries, overall and per device."""
    log = _load_shadow_log()
    resolved_mask = _is_resolved(log)
    resolved = log[resolved_mask].copy()
    if len(resolved) == 0:
        return {"total_resolved": 0, "total_pending": int((~resolved_mask).sum()),
                "overall_agreement_rate": None, "by_device": [],
                "message": "No resolved shadow entries yet."}

    resolved["agreement"] = resolved["agreement"].astype(str).str.lower() == "true"
    overall_rate = round(resolved["agreement"].mean() * 100, 1)

    by_device = resolved.groupby("predicted_device").agg(
        n=("agreement", "count"), agreement_rate=("agreement", "mean"),
    ).reset_index()
    by_device["agreement_rate"] = (by_device["agreement_rate"] * 100).round(1)

    return {
        "total_resolved": int(len(resolved)),
        "total_pending": int((~resolved_mask).sum()),
        "overall_agreement_rate": overall_rate,
        "by_device": by_device.to_dict(orient="records"),
    }


@app.get("/stats")
def stats():
    df = get_events()
    tagged = df[df["primary_device"].notna()]

    # Devices below MIN_CLASS_COUNT are listed separately and excluded
    # from the top-contributor rankings.
    device_counts = tagged["primary_device"].value_counts()
    reliable_devices = device_counts[device_counts >= config.MIN_CLASS_COUNT].index
    low_sample_devices = device_counts[device_counts < config.MIN_CLASS_COUNT].index

    tagged_reliable = tagged[tagged["primary_device"].isin(reliable_devices)]
    tagged_low_sample = tagged[tagged["primary_device"].isin(low_sample_devices)]

    def build_pareto(subset):
        p = subset.groupby("primary_device").agg(
            total_minutes=("mins", "sum"), avg_minutes=("mins", "mean"),
            max_minutes=("mins", "max"), events=("mins", "count"),
        ).reset_index().rename(columns={"primary_device": "device"})
        p["avg_minutes"] = p["avg_minutes"].round(1)
        return p.sort_values("total_minutes", ascending=False)

    pareto = build_pareto(tagged_reliable)
    pareto_low_sample = build_pareto(tagged_low_sample) if len(tagged_low_sample) else pareto.iloc[0:0]

    fmea = load_fmea_scores()
    fused = fuse_pareto_with_fmea(pareto, fmea)
    fused["recommendation"] = fused["device"].apply(get_recommendation)
    fused["low_sample"] = False

    if len(pareto_low_sample):
        fused_low = fuse_pareto_with_fmea(pareto_low_sample, fmea)
        fused_low["recommendation"] = fused_low["device"].apply(get_recommendation)
        fused_low["low_sample"] = True
    else:
        fused_low = pareto_low_sample

    total_events = len(df)
    tagged_events = len(tagged)
    coverage_pct = round(100 * tagged_events / total_events, 1) if total_events else 0.0

    top_volume = pareto.iloc[0] if len(pareto) else None
    top_severity_row = pareto.sort_values("avg_minutes", ascending=False).iloc[0] if len(pareto) else None

    return {
        "site": config.HOME_SITE,
        "total_field_device_delay_minutes": float(tagged_reliable["mins"].sum()),
        "total_tagged_events": int(tagged_events),
        "total_events": int(total_events),
        "unresolved_events": int(total_events - tagged_events),
        "coverage_pct": coverage_pct,
        "top_contributor_volume": {
            "device": top_volume["device"], "total_minutes": float(top_volume["total_minutes"]),
            "events": int(top_volume["events"]),
        } if top_volume is not None else None,
        "top_contributor_severity": {
            "device": top_severity_row["device"], "avg_minutes": float(top_severity_row["avg_minutes"]),
        } if top_severity_row is not None else None,
        "months_covered": config.sort_months(df["month"].dropna().unique()),
        "pareto_by_device": fused.to_dict(orient="records"),
        "pareto_low_sample_devices": fused_low.to_dict(orient="records") if len(fused_low) else [],
    }


@app.get("/device_area_breakdown")
def device_area_breakdown(top_n: int = Query(20, ge=1, le=200)):
    """Device x area cross-tab by total delay minutes."""
    df = get_events()
    tagged = df[df["primary_device"].notna() & df["area"].notna()]
    cross = tagged.groupby(["primary_device", "area"]).agg(
        events=("mins", "count"), total_minutes=("mins", "sum"), avg_minutes=("mins", "mean"),
    ).reset_index().rename(columns={"primary_device": "device"})
    cross["avg_minutes"] = cross["avg_minutes"].round(1)
    cross = cross.sort_values("total_minutes", ascending=False).head(top_n)
    return {"top_device_area_combinations": cross.to_dict(orient="records")}


@app.get("/forecast")
def forecast(device: Optional[str] = Query(None), months: int = Query(4, ge=1, le=12)):
    df = get_events()
    if device:
        device = device.upper()
        if device not in df["primary_device"].dropna().unique():
            raise HTTPException(status_code=400, detail=f"Unknown device '{device}'.")
    return forecast_next_month(df, device=device, forecast_months=months)


@app.get("/forecast/all_devices")
def forecast_all(months: int = Query(4, ge=1, le=12)):
    df = get_events()
    return forecast_all_devices(df, forecast_months=months)


@app.get("/events")
def events(
    device: Optional[str] = Query(None),
    area: Optional[str] = Query(None),
    month: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    site: str = Query(config.HOME_SITE, description="COMBI_MILL (default), ALL, or a plant name"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    site = site.upper()
    if site == config.HOME_SITE:
        df = get_events().copy()
    else:
        df = load_all_events()
        if site != "ALL":
            df = df[df["site"].astype(str).str.upper() == site]
    if device:
        df = df[df["primary_device"] == device.upper()]
    if area:
        df = df[df["area"].astype(str).str.upper() == area.upper()]
    if month:
        df = df[df["month"] == month]
    if search:
        df = df[df["reason_text"].str.contains(search, case=False, na=False)]

    total = len(df)
    df = df.assign(_d=pd.to_datetime(df["date"], errors="coerce")) \
        .sort_values("_d", ascending=False, na_position="last")
    page = df.iloc[offset: offset + limit]

    records = page[["date", "month", "mins", "reason_text", "primary_device", "area", "site", "plant_type"]] \
        .rename(columns={"primary_device": "device"}) \
        .fillna({"device": "UNRESOLVED", "area": "UNKNOWN"})
    records["date"] = records["date"].astype(str)
    records = records.astype(object).where(records.notna(), None).to_dict(orient="records")
    return {"total": total, "limit": limit, "offset": offset, "site": site, "events": records}


@app.get("/devices")
def devices():
    return {"target_devices": config.TARGET_DEVICES}


@app.get("/health_score")
def health_score(top_n: int = Query(10, ge=1, le=50)):
    """Device x area recurrence-risk ranking, plus the Cox model's 14-day
    recurrence probability when it can be fit."""
    df = _load_events_for_health_score()
    df = build_recurrence_features(df)

    scores = compute_simple_health_scores(df)
    top = scores.head(top_n)

    survival_available = len(df) >= MIN_EVENTS_FOR_SURVIVAL_MODEL
    cph = fit_survival_model(df) if survival_available else None

    records = []
    for _, row in top.iterrows():
        rec = {
            "device": row["primary_device"],
            "area": row["area"],
            "health_score": float(row["health_score"]),
            "days_since_last_event": int(row["days_since_last_event_asof_cutoff"]),
            "events_last_90d": int(row["event_count_90d"]),
            "total_events": int(row["total_events"]),
            "low_sample_warning": bool(row["low_sample_warning"]),
        }
        if cph is not None:
            risk = predict_recurrence_risk(
                cph, row["primary_device"], row["area"],
                days_since_last=0, event_count_90d=row["event_count_90d"], horizon_days=14
            )
            rec["survival_model_14d_recurrence_pct"] = risk.get("recurrence_probability_pct")
        records.append(rec)

    return {
        "top_at_risk": records,
        "survival_model_available": cph is not None,
        "note": None if cph is not None else (
            "Survival model not available -- either lifelines is not installed, or there isn't "
            f"enough data yet (need >= {MIN_EVENTS_FOR_SURVIVAL_MODEL} events). "
            "The health_score ranking above is unaffected and remains valid."
        ),
    }


# ------------------------------------------------- Fleet-wide insights --
@app.get("/sites")
def sites():
    return fleet_insights.sites_summary()


@app.get("/fleet/overview")
def fleet_overview():
    return fleet_insights.fleet_overview()


@app.get("/fleet/device_ranking")
def fleet_device_ranking():
    return fleet_insights.device_ranking()


@app.get("/fleet/failure_modes")
def fleet_failure_modes(device: Optional[str] = Query(None)):
    return fleet_insights.failure_modes(device)


@app.get("/fleet/lessons")
def fleet_lessons():
    return fleet_insights.cross_site_lessons()


@app.get("/fleet/trend")
def fleet_trend(top_n_devices: int = Query(6, ge=1, le=16)):
    return fleet_insights.yearly_trend(top_n_devices)


@app.get("/fleet/grid")
def fleet_grid(top_n: int = Query(10, ge=1, le=50)):
    return fleet_insights.grid_insights(top_n)


@app.get("/fleet/plant_reliability")
def fleet_plant_reliability(top_n: int = Query(15, ge=1, le=100)):
    return fleet_insights.plant_reliability(top_n)


@app.get("/power_plants")
def power_plants(country: Optional[str] = Query(None, description="ISO3 code, e.g. IND, USA"),
                 top_n: int = Query(10, ge=1, le=50)):
    return fleet_insights.power_plants_summary(country, top_n)


# Dashboard, mounted last so the API routes take precedence.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
