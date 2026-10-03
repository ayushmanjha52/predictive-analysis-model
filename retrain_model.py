"""
Retraining with rollback.

Runs the standard training pipeline (src/train.py) and adds:
  - timestamped snapshots of the deployed artifacts in models/archive/,
  - a regression guard: if the new model's backtest accuracy is
    lower than the currently deployed model's, the previous artifacts
    are restored (override with allow_regression=True).

Usage: python retrain_model.py
"""
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

from src.config import MODELS_DIR
from src.train import train_and_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS = ["cause_classifier_model.pkl", "tfidf_vectorizer.pkl",
             "feature_names.pkl", "training_manifest.json"]


def _load_manifest():
    path = MODELS_DIR / "training_manifest.json"
    return json.load(open(path)) if path.exists() else None


def _backtest_metric(manifest):
    """Pooled backtest accuracy; falls back to the single-month holdout
    for manifests written before the backtest was introduced."""
    return manifest.get("backtest_accuracy", manifest.get("holdout_accuracy"))


def _archive_current_artifacts(timestamp):
    archive_dir = MODELS_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    for fname in ARTIFACTS:
        src = MODELS_DIR / fname
        if src.exists():
            shutil.copy(src, archive_dir / f"{Path(fname).stem}_{timestamp}{Path(fname).suffix}")
    logger.info(f"Archived pre-retrain artifacts to {archive_dir} with timestamp {timestamp}")


def _restore_from_archive(timestamp):
    archive_dir = MODELS_DIR / "archive"
    for fname in ARTIFACTS:
        backup = archive_dir / f"{Path(fname).stem}_{timestamp}{Path(fname).suffix}"
        if backup.exists():
            shutil.copy(backup, MODELS_DIR / fname)
    logger.info("Previous artifacts restored -- the deployed model is unchanged.")


def retrain_model(allow_regression=False):
    logger.info("=" * 60)
    logger.info("STARTING MODEL RETRAINING")
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    previous_manifest = _load_manifest()
    previous_score = _backtest_metric(previous_manifest) if previous_manifest else None
    if previous_manifest:
        logger.info(f"Currently deployed model: {previous_manifest.get('winner_model')} "
                    f"(backtest accuracy: {previous_score})")
    else:
        logger.info("No existing training_manifest.json -- first-time train.")

    _archive_current_artifacts(timestamp)
    train_and_evaluate()

    new_manifest = _load_manifest()
    if new_manifest is None:
        logger.error("Training finished but wrote no training_manifest.json. Restoring previous artifacts.")
        _restore_from_archive(timestamp)
        return {"status": "failed"}

    new_score = _backtest_metric(new_manifest)
    logger.info(f"New model: {new_manifest.get('winner_model')} (backtest accuracy: {new_score})")

    if previous_score is not None and new_score is not None and new_score < previous_score:
        if not allow_regression:
            logger.warning(f"REGRESSION: new backtest accuracy {new_score} < deployed {previous_score}. "
                           f"Restoring previous artifacts (use allow_regression=True to override).")
            _restore_from_archive(timestamp)
            return {"status": "regression_blocked", "previous": previous_score, "attempted": new_score}

    for fname in ARTIFACTS:
        src = MODELS_DIR / fname
        if src.exists():
            shutil.copy(src, MODELS_DIR / "archive" / f"{Path(fname).stem}_{timestamp}_accepted{Path(fname).suffix}")

    logger.info("Retraining complete. New model deployed.")
    return {"status": "deployed", "previous": previous_score, "new": new_score,
            "winner": new_manifest.get("winner_model"), "timestamp": timestamp}


if __name__ == "__main__":
    retrain_model()
