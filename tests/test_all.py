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