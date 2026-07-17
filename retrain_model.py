"""
Retraining Pipeline for Combi Mill PdM Model
Run this script whenever you have new delay data to update the model.

FIXES vs. previous version -- both confirmed with real data:

1. CRITICAL: used a random train_test_split(test_size=0.20) to evaluate
   the retrained model. Confirmed on this data: this reports 83.6%
   "accuracy" vs. the honest ~62-65% established repeatedly elsewhere in
   this project via proper time-based holdout (train on all-but-latest
   month, test only on that unseen month). A random split shuffles
   similar-phrased delays from every month together, the same way
   k-fold CV did earlier -- it flatters the number without telling you
   anything about performance on a genuinely new future month.

2. CRITICAL: reimplemented its own training logic from scratch (fixed
   RandomForest hyperparameters, no model comparison, no compound-label
   collapsing) instead of reusing train.py's actual methodology. This
   meant every retrain would SILENTLY REPLACE whatever model
   train.py's rigorous comparison had selected as the genuine best
   (currently Logistic Regression, per training_manifest.json) with a
   plain, untuned Random Forest -- using an inflated accuracy number as
   false reassurance that this was an improvement.

3. Never touched training_manifest.json, so after running this script,
   the manifest (which app.py's /model_info reads) would describe a
   model that's no longer actually deployed -- reintroducing the exact
   "stale metadata" bug fixed in app.py earlier this session, via a
   completely different code path.

FIX: this is now a thin wrapper around train.py's train_and_evaluate(),
which already does model comparison + compound-label collapsing + honest
holdout selection + manifest writing correctly. Retraining should mean
"re-run the same rigorous process on updated data," never a separate,
looser process. This script adds ONLY two things on top: timestamped
version snapshots (for rollback/audit) and a regression guard that
refuses to silently keep a new model that's WORSE than what's currently
deployed, on the metric that actually matters (honest holdout accuracy).
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


def _load_manifest():
    path = MODELS_DIR / "training_manifest.json"
    return json.load(open(path)) if path.exists() else None


def _archive_current_artifacts(timestamp):
    """Snapshot the artifacts that were deployed BEFORE this retrain,
    so there's always a way back if the new model regresses."""
    archive_dir = MODELS_DIR / "archive"
    archive_dir.mkdir(exist_ok=True)
    for fname in ["cause_classifier_model.pkl", "tfidf_vectorizer.pkl",
                  "feature_names.pkl", "training_manifest.json"]:
        src = MODELS_DIR / fname
        if src.exists():
            shutil.copy(src, archive_dir / f"{Path(fname).stem}_{timestamp}{Path(fname).suffix}")
    logger.info(f"Archived pre-retrain artifacts to {archive_dir} with timestamp {timestamp}")


def retrain_model(allow_regression=False):
    """
    allow_regression: if False (default), refuses to leave a retrained
    model deployed if its honest holdout accuracy is worse than the
    previously-deployed model's -- restores the pre-retrain artifacts
    from the archive instead. Set True only if you deliberately want to
    accept a regression (e.g. testing, or a known short-term dip while
    gathering more data for a new device).
    """
    logger.info("=" * 60)
    logger.info("STARTING MODEL RETRAINING")
    logger.info("=" * 60)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    previous_manifest = _load_manifest()
    previous_holdout = None
    previous_winner = None
    if previous_manifest:
        # FIX: confirmed by actually running this and inspecting the real
        # manifest -- train.py writes "winner_model" (not "winner"), and
        # "holdout_accuracy" is a TOP-LEVEL field (not nested inside
        # all_candidates[winner]). The old lookup silently returned None
        # for both, every single time, making the regression guard a
        # no-op that never actually compared anything.
        previous_winner = previous_manifest.get("winner_model")
        previous_holdout = previous_manifest.get("holdout_accuracy")
        logger.info(f"Currently deployed model: {previous_winner}  "
                    f"(honest holdout accuracy: {previous_holdout})")
    else:
        logger.info("No existing training_manifest.json found -- this looks like a first-time train.")

    _archive_current_artifacts(timestamp)

    # FIX: reuse train.py's actual rigorous methodology, don't
    # reimplement a separate, looser version of it here.
    logger.info("Running train.py's train_and_evaluate() "
                "(model comparison + honest holdout selection)...")
    # FIX: confirmed by actually running this end-to-end -- train_and_evaluate()
    # in this project's src/train.py takes NO arguments. Class-count
    # filtering happens internally via config.MIN_CLASS_COUNT, not a
    # parameter passed in from here. The old call signature
    # (min_samples_per_class=...) was written against a different
    # version of train.py and crashed with a TypeError the first time
    # this script was actually run against the current codebase.
    train_and_evaluate()

    new_manifest = _load_manifest()
    if new_manifest is None:
        # Genuine failure case: train_and_evaluate() ran without raising,
        # but no manifest was written -- treat this as a hard failure
        # rather than silently continuing with new_winner/new_holdout as
        # None, which would make the regression guard below a no-op
        # exactly like the bug already found and fixed once in this file.
        logger.error("Training completed but no training_manifest.json was found afterward. "
                     "Restoring previous artifacts -- refusing to leave an undefined model deployed.")
        _restore_from_archive(timestamp)
        return

    new_winner = new_manifest.get("winner_model")
    new_holdout = new_manifest.get("holdout_accuracy")
    logger.info(f"New model after retraining: {new_winner}  "
                f"(honest holdout accuracy: {new_holdout})")

    # FIX: regression guard -- never silently deploy a worse model.
    if previous_holdout is not None and new_holdout is not None:
        if new_holdout < previous_holdout and not allow_regression:
            logger.warning(
                f"REGRESSION DETECTED: new model's holdout accuracy ({new_holdout}) "
                f"is WORSE than the previously deployed model's ({previous_holdout}). "
                f"Restoring previous artifacts -- the retrain's output will NOT be "
                f"left deployed. Re-run with allow_regression=True to override "
                f"this (not recommended unless you have a specific reason)."
            )
            _restore_from_archive(timestamp)
            return {"status": "regression_blocked", "previous_holdout": previous_holdout,
                    "attempted_holdout": new_holdout}
        elif new_holdout >= previous_holdout:
            logger.info(f"Improvement or no regression confirmed "
                        f"({previous_holdout} -> {new_holdout}). Keeping new model deployed.")

    # Versioned snapshot of the (accepted) new artifacts, for audit history.
    for fname in ["cause_classifier_model.pkl", "tfidf_vectorizer.pkl",
                  "feature_names.pkl", "training_manifest.json"]:
        src = MODELS_DIR / fname
        if src.exists():
            shutil.copy(src, MODELS_DIR / "archive" / f"{Path(fname).stem}_{timestamp}_accepted{Path(fname).suffix}")

    logger.info("Retraining completed successfully! New model is deployed.")
    return {"status": "deployed", "previous_holdout": previous_holdout, "new_holdout": new_holdout,
            "winner": new_winner, "timestamp": timestamp}


def _restore_from_archive(timestamp):
    archive_dir = MODELS_DIR / "archive"
    for fname in ["cause_classifier_model.pkl", "tfidf_vectorizer.pkl",
                  "feature_names.pkl", "training_manifest.json"]:
        backup = archive_dir / f"{Path(fname).stem}_{timestamp}{Path(fname).suffix}"
        if backup.exists():
            shutil.copy(backup, MODELS_DIR / fname)
    logger.info("Previous artifacts restored -- deployed model is unchanged from before this retrain attempt.")


if __name__ == "__main__":
    retrain_model()