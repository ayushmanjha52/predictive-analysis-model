"""
FastAPI backend -- Combi Mill Field-Device Predictive Maintenance.

Endpoints:
  GET  /health          - liveness + model-loaded check
  GET  /model_info        - manifest: winning model, honest holdout accuracy, classes
  POST /predict            - classify new delay text (+ optional area), with expected duration
  POST /log_event           - classify AND record as a real, timestamped event (real-time)
  GET  /stats               - area-aware Pareto + FMEA fusion + recommendations
  GET  /forecast             - per-device 3-4 month volume forecast, ranked
  GET  /events                - paginated/filterable event log
  GET  /device_area_breakdown  - device x area cross-tab (the specific
                                  "which device, in which area" requirement)

Run: uvicorn app:app --reload --port 8000
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

import json
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from data_loader import load_master_events
from fmea_risk import load_fmea_scores, fuse_pareto_with_fmea, get_recommendation
from forecasting import forecast_next_month, forecast_all_devices
from predict import get_predictor

app = FastAPI(
    title="Combi Mill Field-Device Predictive Maintenance API",
    version="2.0.0",
    description="Classifies delay causes (device + area), forecasts delay volume, "
                "and fuses empirical data with FMEA risk to support a 50% delay reduction target.",
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


# --------------------------------------------------------------- Endpoints --
@app.get("/")
def root():
    return {"message": "Combi Mill Predictive Maintenance API", "docs": "/docs", "health": "/health"}


@app.get("/area_risk")
def area_risk(top_n: int = Query(5, ge=1, le=50)):
    """Which areas currently carry the most delay -- the 'red zone'
    areas requested: ranked by total minutes, tagged field-device
    events only."""
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
    """Which areas carry the most delay minutes that ISN'T yet resolved
    to a specific field device -- turns the coverage gap into an
    actionable target (investigate these areas' logging/tagging) rather
    than a caveat about the dashboard's own limitations."""
    df = get_events()
    unresolved = df[df["primary_device"].isna() & df["area"].notna()]
    hotspots = unresolved.groupby("area").agg(
        total_minutes=("mins", "sum"), events=("mins", "count"),
    ).reset_index().sort_values("total_minutes", ascending=False).head(top_n)
    return {"top_unresolved_hotspots": hotspots.to_dict(orient="records")}


@app.get("/review_queue")
def review_queue(limit: int = Query(50, ge=1, le=500)):
    """Individual unresolved delay events, ranked by minutes (highest
    impact first) -- a queue for a human to review and manually tag.
    Only ~1-2% of unresolved delays were found to contain a safely
    auto-taggable device signal (verified against this project's own
    historical tagging -- see chat history for the analysis); the rest
    genuinely need human judgment, so this endpoint prioritizes a
    reviewer's time rather than attempting to guess.

    FIX: confirmed via production traceback that 9 rows have NaN date
    and 2 rows have NaN reason_text among unresolved delays -- NaN is
    not valid JSON (same class of bug found earlier in fmea_risk.py).
    The previous fillna() only covered 'area', missing these two
    columns entirely, which crashed every real call with
    ValueError: Out of range float values are not JSON compliant: nan.
    """
    df = get_events()
    unresolved = df[df["primary_device"].isna()].copy()
    unresolved = unresolved.sort_values("mins", ascending=False).head(limit)
    records = unresolved[["date", "month", "mins", "reason_text", "area"]].fillna(
        {"area": "UNKNOWN", "date": "", "month": "", "reason_text": "(no description)"}
    ).to_dict(orient="records")
    return {
        "total_unresolved": int(df["primary_device"].isna().sum()),
        "total_unresolved_minutes": float(df[df["primary_device"].isna()]["mins"].sum()),
        "queue": records,
    }


@app.get("/health")
def health():
    model_ready = config.MODEL_PATH.exists() and config.VECTORIZER_PATH.exists()
    return {"status": "ok", "model_ready": model_ready}


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
        result = predictor.predict(req.reason_text, area=req.area)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    result["recommendation"] = get_recommendation(result["predicted_device"])
    return result


@app.post("/log_event")
def log_event(req: PredictRequest):
    """Classify AND record as a real event with a server timestamp --
    appends to master_events.csv and invalidates the cache so /stats,
    /forecast, and /events reflect it on the very next call."""
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
# Records the model's prediction and a maintenance reviewer's independent
# actual diagnosis for the SAME real delay, so agreement rate can be
# measured before this system drives unsupervised maintenance decisions.
# Two-step, deliberately: the prediction is logged blind (shadow_predict),
# then the reviewer's real finding is recorded separately (shadow_resolve)
# -- the reviewer should determine the actual cause independently, not by
# reading the model's guess first, or the comparison means nothing.

class ShadowPredictRequest(BaseModel):
    reason_text: str = Field(..., min_length=3, max_length=2000)
    area: Optional[str] = None
    mins: Optional[float] = Field(None, ge=0)


class ShadowResolveRequest(BaseModel):
    shadow_id: str = Field(..., description="The id returned by /shadow_predict")
    actual_device: str = Field(..., description="What the reviewer actually found -- their independent diagnosis")
    reviewer_note: Optional[str] = Field(None, max_length=1000)


def _load_shadow_log() -> pd.DataFrame:
    cols = ["shadow_id", "logged_at", "reason_text", "area", "mins",
            "predicted_device", "confidence", "actual_device",
            "resolved_at", "reviewer_note", "agreement"]
    if not config.SHADOW_LOG_PATH.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(config.SHADOW_LOG_PATH)


@app.post("/shadow_predict")
def shadow_predict(req: ShadowPredictRequest):
    """Step 1: log the model's prediction for a real delay, without
    revealing it to the reviewer yet. Returns a shadow_id -- the
    reviewer independently determines the actual cause through normal
    means, then submits it via /shadow_resolve using this id."""
    predictor = get_predictor_safe()
    now = datetime.now()
    result = predictor.predict(req.reason_text, area=req.area)

    import uuid
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

    # Deliberately does NOT return the prediction in this response --
    # the point of shadow mode is the reviewer's diagnosis stays
    # independent of the model's guess until both are recorded.
    return {"shadow_id": shadow_id, "logged": True,
            "message": "Prediction recorded. Determine the actual cause independently, then submit via /shadow_resolve."}


@app.post("/shadow_resolve")
def shadow_resolve(req: ShadowResolveRequest):
    """Step 2: record the reviewer's independent actual diagnosis and
    compute whether it agreed with the model's (now-revealed) prediction."""
    log = _load_shadow_log()
    match = log["shadow_id"] == req.shadow_id
    if not match.any():
        raise HTTPException(status_code=404, detail=f"No shadow entry found for id '{req.shadow_id}'.")

    now = datetime.now()
    idx = log[match].index[0]
    predicted = str(log.loc[idx, "predicted_device"]).upper()
    actual = req.actual_device.upper().strip()
    agreement = predicted == actual

    # FIX: confirmed by actually running this end-to-end (not just code
    # review) -- writing a string into a column that started as all
    # empty strings/NaN raises pandas.errors.LossySetitemError /
    # TypeError, because pandas infers an incompatible dtype (often
    # float64) for a column with no non-null values yet on first
    # write/read-back from CSV. Casting the whole column to object
    # dtype before assignment avoids this.
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
    """Running agreement rate across all resolved shadow-mode entries --
    the core metric for deciding whether this system is ready to drive
    unsupervised decisions. Returns per-device breakdown too, since
    overall accuracy can hide a weak class (Encoder has been the
    consistently weakest class across every model tried in this
    project)."""
    log = _load_shadow_log()
    resolved = log[log["resolved_at"] != ""].copy()
    if len(resolved) == 0:
        return {"total_resolved": 0, "overall_agreement_rate": None,
                "by_device": [], "message": "No resolved shadow entries yet."}

    resolved["agreement"] = resolved["agreement"].astype(str).str.lower() == "true"
    overall_rate = round(resolved["agreement"].mean() * 100, 1)

    by_device = resolved.groupby("predicted_device").agg(
        n=("agreement", "count"), agreement_rate=("agreement", "mean"),
    ).reset_index()
    by_device["agreement_rate"] = (by_device["agreement_rate"] * 100).round(1)

    return {
        "total_resolved": int(len(resolved)),
        "total_pending": int((log["resolved_at"] == "").sum()),
        "overall_agreement_rate": overall_rate,
        "by_device": by_device.to_dict(orient="records"),
    }


@app.get("/stats")
def stats():
    df = get_events()
    tagged = df[df["primary_device"].notna()]

    # FIX: RFID, TT, HIP (each n=1 in the current dataset) were still
    # counted in the Pareto/severity ranking despite being dropped from
    # classifier training (too few examples per config.MIN_CLASS_COUNT).
    # A single event's avg_minutes is not a meaningful "severity" signal
    # and could win Top Contributor (Severity) purely by chance on one
    # data point. Split into "reliable" (>= MIN_CLASS_COUNT samples,
    # shown in the main Pareto/severity ranking) vs "low_sample" (shown
    # separately, flagged, never used for Top Contributor rankings).
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
        "months_covered": sorted(df["month"].dropna().unique().tolist(), key=config.month_sort_key),
        "pareto_by_device": fused.to_dict(orient="records"),
        "pareto_low_sample_devices": fused_low.to_dict(orient="records") if len(fused_low) else [],
    }


@app.get("/device_area_breakdown")
def device_area_breakdown(top_n: int = Query(20, ge=1, le=200)):
    """Device x Area cross-tab -- 'which device, in which area, is worst'."""
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
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    df = get_events().copy()
    if device:
        df = df[df["primary_device"] == device.upper()]
    if area:
        df = df[df["area"].str.upper() == area.upper()]
    if month:
        df = df[df["month"] == month]
    if search:
        df = df[df["reason_text"].str.contains(search, case=False, na=False)]

    total = len(df)
    df = df.sort_values("date", ascending=False, na_position="last")
    page = df.iloc[offset: offset + limit]

    records = page[["date", "month", "mins", "reason_text", "primary_device", "area"]].rename(
        columns={"primary_device": "device"}
    ).fillna({"device": "UNRESOLVED", "area": "UNKNOWN"}).to_dict(orient="records")

    return {"total": total, "limit": limit, "offset": offset, "events": records}


@app.get("/devices")
def devices():
    return {"target_devices": config.TARGET_DEVICES}