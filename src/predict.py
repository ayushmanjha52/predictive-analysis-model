"""
CausePredictor -- the ONE prediction implementation, used by both the
API and any CLI/test. Never reimplemented elsewhere (that duplication
caused repeated production bugs earlier in this project).

Returns, per your requirement: predicted device, confidence, AND the
typical/expected delay duration for that device (from historical data)
-- so a classification isn't just "what" but also "how long this
usually takes."
"""
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from typing import Optional

import joblib
import numpy as np
import pandas as pd

import config
from feature_engineering import prepare_modeling_data, align_features
from data_loader import load_master_events


# Explicit device-name vocabulary for the home site. Abbreviations
# (PH/PS/FS/PX) are the plant's own Sub_Category codes (see ingest.py's
# DEVICE_NORMALIZE). When a delay text names EXACTLY ONE device class,
# that is far more reliable than any learned pattern -- measured on the
# home site's labeled data: ~29% of events name exactly one device, and
# the named device matches the human label ~94% of the time. It mainly
# fixes rare classes the model has barely seen (e.g. "flow switch
# changed" was predicted PRESSURE_SWITCH because FLOW_SWITCH had ~5
# training examples). Whether it is used at all is decided by train.py's
# honest backtest, not assumed.
EXPLICIT_DEVICE_NAMES = {
    "PHOTOCELL": r"photo ?cells?|\bph\b|photo ?sensors?",
    "HMD": r"\bhmds?\b|hot metal detector",
    "PROXIMITY": r"\bprox(?:imity|y)?\b|\bpx\b",
    "LVDT": r"\blvdts?\b",
    "ENCODER": r"\bencoders?\b",
    "PRESSURE_SWITCH": r"pressure switch(?:es)?|\bps\b",
    "FLOW_SWITCH": r"flow switch(?:es)?|\bfs\b",
    "LASER": r"\blasers?\b",
}
_EXPLICIT = {k: re.compile(v, re.I) for k, v in EXPLICIT_DEVICE_NAMES.items()}


def explicit_device_mention(text) -> Optional[str]:
    found = {k for k, rx in _EXPLICIT.items() if rx.search(str(text or ""))}
    return found.pop() if len(found) == 1 else None


def apply_explicit_mention_rule(texts, preds, allowed):
    """Override model predictions where the text names exactly one
    allowed device class."""
    out = list(preds)
    for i, t in enumerate(texts):
        named = explicit_device_mention(t)
        if named and (not allowed or named in allowed):
            out[i] = named
    return np.asarray(out, dtype=object)


def restrict_to_classes(proba, classes, allowed):
    """Zero out (and renormalize away) classes not in `allowed`. Used at
    the home site so a Combi Mill delay is never labeled with a device
    class that only exists in the power-plant fleet data (e.g.
    LEVEL_TRANSMITTER). Applied identically in train.py's honest
    backtest, so the reported accuracy includes this rule."""
    proba = np.asarray(proba, dtype=float)
    if not allowed:
        return proba
    mask = np.array([c in allowed for c in classes], dtype=float)
    out = proba * mask
    totals = out.sum(axis=1, keepdims=True)
    return np.where(totals > 0, out / np.where(totals > 0, totals, 1), proba)


class CausePredictor:
    def __init__(self):
        for path, label in [(config.MODEL_PATH, "model"), (config.VECTORIZER_PATH, "vectorizer"),
                             (config.FEATURE_NAMES_PATH, "feature metadata")]:
            if not path.exists():
                raise FileNotFoundError(
                    f"{label} not found at {path}. Run `python src/train.py` first."
                )
        self.model = joblib.load(config.MODEL_PATH)
        self.vectorizer = joblib.load(config.VECTORIZER_PATH)
        meta = joblib.load(config.FEATURE_NAMES_PATH)
        self.feature_names = meta["feature_names"]
        self.known_areas = meta["known_areas"]
        self.home_classes = set(meta.get("home_classes") or [])
        self.use_explicit_rule = bool(meta.get("explicit_mention_rule", False))
        self.explicit_rule_precision = meta.get("explicit_rule_precision")

        # Precompute typical duration per device from historical data,
        # so a prediction can say "Photocell delays typically run ~23
        # min" alongside the classification itself.
        self._device_duration_stats = self._compute_duration_stats()

    def _compute_duration_stats(self):
        try:
            df = load_master_events()
            tagged = df[df["primary_device"].notna()]
            stats = tagged.groupby("primary_device")["mins"].agg(["mean", "median", "max", "count"])
            return stats.to_dict(orient="index")
        except Exception:
            return {}

    def predict(self, reason_text: str, area: Optional[str] = None, top_n: int = 3,
                plant_type: Optional[str] = None) -> dict:
        """plant_type defaults to the home site (Combi Mill). Pass e.g.
        'NUCLEAR_POWER' to classify a delay from a power plant -- then
        every device class the model knows is allowed."""
        plant_type = (plant_type or config.HOME_PLANT_TYPE).upper()
        df = pd.DataFrame({"reason_text": [reason_text], "area": [area], "plant_type": [plant_type]})
        X, _, _ = prepare_modeling_data(df, vectorizer=self.vectorizer, fit=False,
                                          known_areas=self.known_areas)
        X = align_features(X, self.feature_names)

        classes = self.model.classes_
        proba = self.model.predict_proba(X)
        if plant_type == config.HOME_PLANT_TYPE:
            proba = restrict_to_classes(proba, classes, self.home_classes)
        proba = proba[0]
        top_idx = np.argsort(proba)[-top_n:][::-1]

        candidates = [{"device": classes[i], "confidence": round(float(proba[i]), 4)} for i in top_idx]
        top_device = candidates[0]["device"]
        top_confidence = candidates[0]["confidence"]
        decision_basis = "model"

        named = explicit_device_mention(reason_text) if (
            self.use_explicit_rule and plant_type == config.HOME_PLANT_TYPE) else None
        if named and named in self.home_classes:
            # Confidence = the rule's MEASURED precision on labeled home
            # data (from training), not a made-up number.
            decision_basis = "explicit_device_mention"
            top_device = named
            top_confidence = round(max(self.explicit_rule_precision or 0.0,
                                       float(proba[list(classes).index(named)])), 4)
            candidates = [{"device": named, "confidence": top_confidence}] + \
                         [c for c in candidates if c["device"] != named][:top_n - 1]
        low_confidence = top_confidence < config.CONFIDENCE_THRESHOLD

        duration_info = self._device_duration_stats.get(top_device, {})

        return {
            "reason_text": reason_text,
            "area": area,
            "plant_type": plant_type,
            "predicted_device": top_device,
            "confidence": top_confidence,
            "decision_basis": decision_basis,
            "low_confidence": low_confidence,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "top_candidates": candidates,
            "typical_delay_minutes": round(duration_info.get("mean", 0), 1) if duration_info else None,
            "median_delay_minutes": round(duration_info.get("median", 0), 1) if duration_info else None,
            "max_observed_delay_minutes": duration_info.get("max") if duration_info else None,
            "based_on_n_historical_events": int(duration_info.get("count", 0)) if duration_info else 0,
        }


_predictor_instance = None


def get_predictor() -> CausePredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = CausePredictor()
    return _predictor_instance


if __name__ == "__main__":
    p = CausePredictor()
    result = p.predict("photocell lens dirty at cold saw", area="SAW")
    import json
    print(json.dumps(result, indent=2))