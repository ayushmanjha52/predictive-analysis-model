"""
Combined test suite. Run with: pytest tests/test_all.py -v
(or: python tests/test_all.py to run without pytest, using the
built-in assertion runner at the bottom)
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import config
from data_loader import load_master_events, primary_device
from feature_engineering import prepare_modeling_data, align_features
from predict import CausePredictor


def test_master_events_exists_and_has_required_columns():
    assert config.MASTER_EVENTS_PATH.exists()
    df = pd.read_csv(config.MASTER_EVENTS_PATH)
    required = {"date", "month", "mins", "reason_text", "field_device", "area"}
    assert required.issubset(set(df.columns))


def test_no_entire_month_missing_dates():
    df = pd.read_csv(config.MASTER_EVENTS_PATH)
    parsed = pd.to_datetime(df["date"], errors="coerce")
    tmp = df.assign(_ok=parsed.notna())
    zero_rate = tmp.groupby("month")["_ok"].mean()
    zero_months = zero_rate[zero_rate == 0].index.tolist()
    assert not zero_months, f"Months with 0% parseable dates: {zero_months}"


def test_compound_labels_collapse_correctly():
    df = pd.read_csv(config.MASTER_EVENTS_PATH)
    for raw in df["field_device"].dropna().unique():
        collapsed = primary_device(raw)
        assert collapsed is None or ("/" not in collapsed and "_/_" not in collapsed)


def test_no_non_device_categories_present():
    df = pd.read_csv(config.MASTER_EVENTS_PATH)
    non_device = {"SEQUENCE_BREAK", "OVERTRAVEL", "ROLLER_TABLE"}
    present = set(df["field_device"].dropna().unique()) & non_device
    assert not present, f"Non-device categories leaked into field_device: {present}"


def test_feature_alignment_matches_deployed_model():
    predictor = CausePredictor()
    df = pd.DataFrame({"reason_text": ["hmd lens dirty"], "area": ["BDM"]})
    X, _, _ = prepare_modeling_data(df, vectorizer=predictor.vectorizer, fit=False,
                                      known_areas=predictor.known_areas)
    X_aligned = align_features(X, predictor.feature_names)
    assert list(X_aligned.columns) == predictor.feature_names
    proba = predictor.model.predict_proba(X_aligned)
    assert proba.shape[1] == len(predictor.model.classes_)


def test_mins_excluded_from_features():
    """Regression test: 'mins' was previously accidentally included as
    a numeric feature, causing a NaN crash in Logistic Regression."""
    df = load_master_events()
    labeled = df[df["primary_device"].notna()].copy()
    X, _, _ = prepare_modeling_data(labeled, fit=True)
    assert "mins" not in X.columns


def test_domain_obvious_predictions():
    predictor = CausePredictor()
    cases = [
        ("photocell lens dirty at cold saw", "PHOTOCELL"),
        ("hmd cleaning in bdm", "HMD"),
        ("pressure switch oil low warning", "PRESSURE_SWITCH"),
    ]
    for text, expected in cases:
        result = predictor.predict(text)
        assert result["predicted_device"] == expected, (
            f"Expected {expected} for '{text}', got {result['predicted_device']}"
        )


def test_prediction_includes_duration_context():
    predictor = CausePredictor()
    result = predictor.predict("photocell lens dirty", area="SAW")
    assert "typical_delay_minutes" in result
    assert "based_on_n_historical_events" in result
    assert result["based_on_n_historical_events"] > 0


def test_low_confidence_flag_on_vague_text():
    predictor = CausePredictor()
    flagged = 0
    vague = ["delay occurred", "issue found", "problem noted", "stoppage happened"]
    for text in vague:
        r = predictor.predict(text)
        if r["low_confidence"]:
            flagged += 1
    assert flagged >= len(vague) * 0.5


def test_manifest_reports_holdout_not_just_cv():
    import json
    assert config.MANIFEST_PATH.exists()
    manifest = json.load(open(config.MANIFEST_PATH))
    assert "holdout_accuracy" in manifest
    assert "cv_accuracy" in manifest
    assert manifest["holdout_accuracy"] is not None
    assert manifest.get("backtest_accuracy") is not None


# ---------------------------------------------------- multi-site / v3 --
def test_month_sort_is_chronological_and_open_ended():
    assert config.sort_months(["Jul-26", "Nov-25", "Jan-27", "Apr-26"]) == ["Nov-25", "Apr-26", "Jul-26", "Jan-27"]


def test_live_predictions_never_used_as_training_labels():
    from train import get_labeled_data
    df = get_labeled_data()
    assert not (df["tag_source"] == "live_prediction").any()


def test_external_rows_never_in_backtest_test_set():
    from train import get_labeled_data, get_backtest_months
    df = get_labeled_data()
    months = get_backtest_months(df[df["is_home"]])
    assert months, "no backtest months found"
    for m in months:
        test = df[df["is_home"] & (df["month"] == m)]
        assert (test["site"] == config.HOME_SITE).all()


def test_home_predictions_restricted_to_home_classes():
    predictor = CausePredictor()
    r = predictor.predict("level transmitter failed low causing feedwater trip")
    if predictor.home_classes:
        assert r["predicted_device"] in predictor.home_classes
        assert all(c["device"] in predictor.home_classes for c in r["top_candidates"] if c["confidence"] > 0)


def test_forecast_counts_quiet_months_as_zero():
    from forecasting import _monthly_series, observed_months
    df = load_master_events()
    months = observed_months(df)
    for dev in df["primary_device"].dropna().unique():
        s = _monthly_series(df, device=dev)
        assert s["month"].tolist() == months


def test_external_labels_only_single_device_events():
    if not config.EXTERNAL_EVENTS_PATH.exists():
        return
    from data_loader import load_external_events
    ext = load_external_events()
    labeled = ext[ext["primary_device"].notna()]
    assert len(labeled) > 0
    assert (labeled["n_device_classes_mentioned"] == 1).all()


def test_api_endpoints_return_valid_json():
    from fastapi.testclient import TestClient
    sys.path.append(str(Path(__file__).parent.parent))
    from app import app
    client = TestClient(app)
    for path in ["/health", "/model_info", "/stats", "/forecast/all_devices", "/review_queue",
                 "/shadow_stats", "/health_score", "/sites", "/fleet/overview",
                 "/fleet/device_ranking", "/fleet/lessons", "/fleet/trend", "/fleet/grid",
                 "/fleet/plant_reliability", "/power_plants", "/power_plants?country=IND",
                 "/events?site=ALL&limit=5"]:
        r = client.get(path)
        assert r.status_code == 200, f"{path} -> {r.status_code}: {r.text[:200]}"
    r = client.post("/predict", json={"reason_text": "pressure switch fault at cooling bed", "area": "COOLING_BED"})
    assert r.status_code == 200 and r.json()["predicted_device"]
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
    assert client.get("/config.js").status_code == 200


def test_runtime_requirements_cover_deployed_model():
    """Render installs requirements.txt only; the pickled model's library must be in it."""
    import joblib
    model = joblib.load(config.MODEL_PATH)
    package = {"sklearn": "scikit-learn"}.get(type(model).__module__.split(".")[0],
                                               type(model).__module__.split(".")[0])
    reqs = (Path(__file__).parent.parent / "requirements.txt").read_text().lower()
    assert package.lower() in reqs, f"{package} is needed by the deployed model but missing from requirements.txt"


if __name__ == "__main__":
    import traceback
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")