"""
THE canonical training script. Compares model families, selects the
winner by HONEST TIME-BASED HOLDOUT accuracy (train on all months
except the latest, test only on that unseen month) -- NOT cross-
validation score alone. This project has repeatedly confirmed CV can
overstate real-world performance by 10-25 points on this kind of
monthly-batched data.

Also reports device x area combinations (Section: AREA BREAKDOWN)
since that's now a stated requirement -- "which device, in which area,
is the worst" rather than device alone.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import logging
from datetime import datetime

import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

import config
from data_loader import load_master_events
from feature_engineering import prepare_modeling_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CANDIDATE_MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, max_features="sqrt",
        min_samples_split=4, min_samples_leaf=1,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=10,
        min_samples_leaf=15, random_state=42,
    ),
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
}

try:
    from catboost import CatBoostClassifier
    CANDIDATE_MODELS["catboost"] = CatBoostClassifier(
        iterations=250, depth=6, learning_rate=0.05,
        random_state=42, verbose=False, auto_class_weights="Balanced",
    )
except ImportError:
    logger.info("catboost not installed -- skipping (optional: pip install catboost)")


def get_labeled_data(min_samples_per_class=config.MIN_CLASS_COUNT):
    df = load_master_events()
    df = df[df["primary_device"].notna()].copy()
    counts = df["primary_device"].value_counts()
    rare = counts[counts < min_samples_per_class].index.tolist()
    if rare:
        logger.info(f"Dropping classes with <{min_samples_per_class} examples: {rare}")
    return df[~df["primary_device"].isin(rare)].reset_index(drop=True)


def get_latest_month(months_present):
    ordered = [m for m in config.MONTH_ORDER if m in months_present]
    return ordered[-1] if ordered else sorted(months_present)[-1]


def evaluate_holdout(model, df, latest_month):
    train_df = df[df["month"] != latest_month].reset_index(drop=True)
    test_df = df[df["month"] == latest_month].reset_index(drop=True)
    if len(test_df) < 5:
        return None

    X_train, vec, known_areas = prepare_modeling_data(train_df, fit=True)
    y_train = train_df["primary_device"]

    X_test, _, _ = prepare_modeling_data(test_df, vectorizer=vec, fit=False, known_areas=known_areas)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    y_test = test_df["primary_device"]

    valid_mask = y_test.isin(y_train.unique())
    if valid_mask.sum() == 0:
        return None

    model_clone = type(model)(**model.get_params())
    model_clone.fit(X_train, y_train)
    preds = model_clone.predict(X_test[valid_mask])
    report = classification_report(y_test[valid_mask], preds, zero_division=0, output_dict=True)
    return {"accuracy": report["accuracy"], "n_test": int(valid_mask.sum()), "report": report}


def compare_all_models(df, X, y):
    months_present = df["month"].dropna().unique().tolist()
    latest_month = get_latest_month(months_present)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    logger.info("=== Comparing candidate models (CV accuracy AND honest holdout) ===")
    for name, model in CANDIDATE_MODELS.items():
        try:
            y_pred_cv = cross_val_predict(model, X, y, cv=skf)
            cv_report = classification_report(y, y_pred_cv, zero_division=0, output_dict=True)
            cv_acc = cv_report["accuracy"]
        except Exception as e:
            logger.warning(f"  {name}: CV failed ({e}) -- skipping")
            continue

        holdout = evaluate_holdout(model, df, latest_month)
        holdout_acc = holdout["accuracy"] if holdout else None
        results[name] = {"cv_accuracy": cv_acc, "holdout_accuracy": holdout_acc,
                          "cv_report": cv_report, "holdout_report": holdout["report"] if holdout else None}
        logger.info(f"  {name:24s} CV={cv_acc:.3f}   Holdout({latest_month})={holdout_acc}")

    have_holdout = {k: v for k, v in results.items() if v["holdout_accuracy"] is not None}
    if have_holdout:
        winner = max(have_holdout, key=lambda k: have_holdout[k]["holdout_accuracy"])
        logger.info(f"Winner selected by HOLDOUT accuracy (the honest metric): {winner}")
    else:
        winner = max(results, key=lambda k: results[k]["cv_accuracy"])
        logger.warning(f"No holdout available -- winner selected by CV only: {winner}")

    return winner, CANDIDATE_MODELS[winner], results, latest_month


def report_area_breakdown(df):
    """Device x Area cross-tab -- the specific requirement: not just
    'which device' but 'which device, in which area' is worst."""
    tagged = df[df["primary_device"].notna() & df["area"].notna()]
    cross = tagged.groupby(["primary_device", "area"]).agg(
        events=("mins", "count"), total_minutes=("mins", "sum"), avg_minutes=("mins", "mean"),
    ).reset_index().sort_values("total_minutes", ascending=False)
    cross["avg_minutes"] = cross["avg_minutes"].round(1)
    logger.info("=== Top 10 Device x Area combinations by total delay minutes ===")
    for _, row in cross.head(10).iterrows():
        logger.info(f"  {row['primary_device']:16s} @ {row['area']:20s} "
                    f"{row['total_minutes']:6.0f} min  ({row['events']} events, avg {row['avg_minutes']})")
    cross.to_csv(config.REPORTS_DIR / "device_area_breakdown.csv", index=False)
    return cross


def train_and_evaluate():
    logger.info("Loading labeled data...")
    df = get_labeled_data()
    logger.info(f"{len(df)} labeled events across {df['primary_device'].nunique()} classes")

    report_area_breakdown(df)

    X, vectorizer, known_areas = prepare_modeling_data(df, fit=True)
    y = df["primary_device"]
    logger.info(f"Training on {X.shape[1]} features with {len(X)} samples")

    winner_name, winner_model, all_results, latest_month = compare_all_models(df, X, y)
    winner_result = all_results[winner_name]

    pd.DataFrame(winner_result["cv_report"]).transpose().to_csv(config.CLASSIFICATION_REPORT_PATH)
    if winner_result["holdout_report"]:
        pd.DataFrame(winner_result["holdout_report"]).transpose().to_csv(config.HOLDOUT_REPORT_PATH)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds = cross_val_predict(winner_model, X, y, cv=skf)
    labels = sorted(y.unique())
    cm = confusion_matrix(y, preds, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix (5-Fold CV) - {winner_name}")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=200)
    plt.close()

    logger.info(f"Training final '{winner_name}' model on full data...")
    winner_model.fit(X, y)

    feature_names = X.columns.tolist()
    n_model_features = getattr(winner_model, "n_features_in_", len(feature_names))
    if n_model_features != len(feature_names):
        raise RuntimeError(
            f"Refusing to save: model trained on {n_model_features} features but "
            f"feature_names has {len(feature_names)} entries."
        )

    joblib.dump(winner_model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)
    joblib.dump({"feature_names": feature_names, "known_areas": known_areas}, config.FEATURE_NAMES_PATH)

    manifest = {
        "trained_at": datetime.now().isoformat(),
        "winner_model": winner_name,
        "n_features": len(feature_names),
        "n_samples": len(X),
        "n_classes": int(y.nunique()),
        "classes": sorted(y.unique().tolist()),
        "holdout_month": latest_month,
        "holdout_accuracy": winner_result["holdout_accuracy"],
        "cv_accuracy": winner_result["cv_accuracy"],
        "all_candidates": {
            name: {"cv_accuracy": round(r["cv_accuracy"], 4),
                   "holdout_accuracy": round(r["holdout_accuracy"], 4) if r["holdout_accuracy"] else None}
            for name, r in all_results.items()
        },
    }
    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"FINAL: winner={winner_name}  CV={winner_result['cv_accuracy']:.3f}  "
                f"Holdout={winner_result['holdout_accuracy']}")
    return winner_model


if __name__ == "__main__":
    train_and_evaluate()