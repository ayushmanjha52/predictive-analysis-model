"""
CausePredictor -- the ONE prediction implementation, used by both the
API and any CLI/test. Never reimplemented elsewhere (that duplication
caused repeated production bugs earlier in this project).

Returns, per your requirement: predicted device, confidence, AND the
typical/expected delay duration for that device (from historical data)
-- so a classification isn't just "what" but also "how long this
usually takes."
"""
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

    def predict(self, reason_text: str, area: Optional[str] = None, top_n: int = 3) -> dict:
        df = pd.DataFrame({"reason_text": [reason_text], "area": [area]})
        X, _, _ = prepare_modeling_data(df, vectorizer=self.vectorizer, fit=False,
                                          known_areas=self.known_areas)
        X = align_features(X, self.feature_names)

        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        top_idx = np.argsort(proba)[-top_n:][::-1]

        candidates = [{"device": classes[i], "confidence": round(float(proba[i]), 4)} for i in top_idx]
        top_device = candidates[0]["device"]
        top_confidence = candidates[0]["confidence"]
        low_confidence = top_confidence < config.CONFIDENCE_THRESHOLD

        duration_info = self._device_duration_stats.get(top_device, {})

        return {
            "reason_text": reason_text,
            "area": area,
            "predicted_device": top_device,
            "confidence": top_confidence,
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