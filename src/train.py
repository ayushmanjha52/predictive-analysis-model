"""
Training. Compares model families, training-data mixes and the
explicit-device-mention rule, and selects by a time-based
backtest on the home site's own hand-tagged data.

Training data:
  - Home site (Combi Mill) hand-tagged events: always included.
  - External power-plant events (keyword-labeled, see external_data.py):
    included at a reduced sample weight only if that beats the best
    home-only configuration by MIN_GAIN_FOR_EXTERNAL.

Evaluation (rolling-origin backtest): for each of the last
N_BACKTEST_MONTHS home months with enough labeled events, train on
every earlier home month (plus external data, per config) and test on
that month. External rows are never in a test set. Accuracy is pooled
across the test months; shuffled CV is reported for reference only, as
it overstates performance on monthly-batched data.

Also writes the device x area breakdown report.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import json
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import config
from data_loader import load_master_events, load_external_events
from feature_engineering import prepare_modeling_data
from predict import restrict_to_classes, apply_explicit_mention_rule, explicit_device_mention

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_BACKTEST_MONTHS = 3
MIN_TEST_EVENTS_PER_MONTH = 10

# None = home data only; otherwise the sample weight of each external
# (keyword-labeled) row relative to a hand-tagged home row.
EXTERNAL_WEIGHT_OPTIONS = [None, 0.15, 0.4, 1.0]

# Minimum backtest gain required to adopt external data (with ~160 test
# events, one event is ~0.6 points).
MIN_GAIN_FOR_EXTERNAL = 0.02

CANDIDATE_MODELS = {
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, max_features="sqrt",
        min_samples_split=4, min_samples_leaf=1,
        class_weight="balanced_subsample", random_state=42, n_jobs=-1,
    ),
    "hist_gradient_boosting": HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.08, max_depth=8,
        min_samples_leaf=10, random_state=42,
    ),
    "logistic_regression": LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"),
}

try:
    from catboost import CatBoostClassifier
    CANDIDATE_MODELS["catboost"] = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.08,
        random_state=42, verbose=False, auto_class_weights="Balanced", thread_count=-1,
    )
except ImportError:
    logger.info("catboost not installed -- skipping (optional: pip install catboost)")


def _clone(model):
    try:
        return clone(model)
    except Exception:
        return type(model)(**model.get_params())


def get_labeled_data(min_samples_per_class=config.MIN_CLASS_COUNT):
    """Home + external labeled events, with an is_home flag. A class is
    kept if it has >= min_samples_per_class examples across all sites."""
    home = load_master_events()
    # Rows from /log_event are labeled with the model's own prediction,
    # so they are not training labels.
    home = home[home["primary_device"].notna() & (home["tag_source"] != "live_prediction")].copy()
    home["is_home"] = True

    ext = load_external_events()
    ext = ext[ext["primary_device"].notna()].copy()
    ext["is_home"] = False

    df = pd.concat([home, ext], ignore_index=True, sort=False)
    counts = df["primary_device"].value_counts()
    rare = counts[counts < min_samples_per_class].index.tolist()
    if rare:
        logger.info(f"Dropping classes with <{min_samples_per_class} examples: {rare}")
    df = df[~df["primary_device"].isin(rare)].reset_index(drop=True)
    return df


def get_backtest_months(home_df):
    counts = home_df.groupby("month").size()
    eligible = [m for m in config.sort_months(counts.index) if counts[m] >= MIN_TEST_EVENTS_PER_MONTH]
    # the first month has nothing earlier to train on
    eligible = eligible[1:]
    return eligible[-N_BACKTEST_MONTHS:]


def training_subset(df, ext_weight, before_month=None):
    """Training rows for a config. With before_month, home rows are
    restricted to strictly earlier months; external rows (another site)
    are used regardless of date."""
    home = df[df["is_home"]]
    if before_month is not None:
        key = config.month_sort_key(before_month)
        home = home[home["month"].map(config.month_sort_key) < key]
    parts = [home]
    if ext_weight is not None:
        parts.append(df[~df["is_home"]])
    train = pd.concat(parts).reset_index(drop=True)
    weights = np.where(train["is_home"], 1.0, ext_weight or 0.0)
    return train, weights


def fit_predict(model, train_df, weights, test_df):
    """Returns {False: model predictions, True: predictions with the
    explicit-device-mention rule applied} -- both from ONE fit."""
    X_train, vec, known_areas = prepare_modeling_data(train_df, fit=True)
    X_test, _, _ = prepare_modeling_data(test_df, vectorizer=vec, fit=False, known_areas=known_areas)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    m = _clone(model)
    m.fit(X_train, train_df["primary_device"], sample_weight=weights)
    # As in the deployed predictor: home predictions use home classes only.
    home_classes = set(train_df.loc[train_df["is_home"], "primary_device"])
    proba = restrict_to_classes(m.predict_proba(X_test), m.classes_, home_classes)
    preds = np.asarray(m.classes_, dtype=object)[proba.argmax(axis=1)]
    return {False: preds,
            True: apply_explicit_mention_rule(test_df["reason_text"].tolist(), preds, home_classes)}


def backtest(model, df, ext_weight, months):
    """Rolling-origin backtest on home months. Returns, for rule in
    (False, True): (pooled accuracy, per-month accuracy, y_true, y_pred)."""
    out = {rule: {"y_true": [], "y_pred": [], "per_month": {}} for rule in (False, True)}
    for m in months:
        train_df, w = training_subset(df, ext_weight, before_month=m)
        test_df = df[df["is_home"] & (df["month"] == m)].reset_index(drop=True)
        test_df = test_df[test_df["primary_device"].isin(train_df["primary_device"].unique())] \
            .reset_index(drop=True)
        if len(test_df) == 0:
            continue
        preds = fit_predict(model, train_df, w, test_df)
        truth = test_df["primary_device"].values
        for rule in (False, True):
            out[rule]["per_month"][m] = float((preds[rule] == truth).mean())
            out[rule]["y_true"].extend(truth.tolist())
            out[rule]["y_pred"].extend(preds[rule].tolist())
    result = {}
    for rule, r in out.items():
        acc = float(np.mean(np.array(r["y_true"]) == np.array(r["y_pred"]))) if r["y_true"] else None
        result[rule] = (acc, r["per_month"], r["y_true"], r["y_pred"])
    return result


def home_cv(model, df, ext_weight, rule, n_splits=5):
    """Stratified CV over home rows (external rows, if used, always train).
    Reference only."""
    home = df[df["is_home"]].reset_index(drop=True)
    ext = df[~df["is_home"]]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    correct = 0
    for tr_idx, te_idx in skf.split(home, home["primary_device"]):
        parts = [home.iloc[tr_idx]] + ([ext] if ext_weight is not None else [])
        train_df = pd.concat(parts).reset_index(drop=True)
        w = np.where(train_df["is_home"], 1.0, ext_weight or 0.0)
        preds = fit_predict(model, train_df, w, home.iloc[te_idx].reset_index(drop=True))[rule]
        correct += int((preds == home.iloc[te_idx]["primary_device"].values).sum())
    return correct / len(home)


def compare_all(df, months):
    n_ext = int((~df["is_home"]).sum())
    weight_options = EXTERNAL_WEIGHT_OPTIONS if n_ext else [None]
    results = []
    logger.info(f"=== Comparing models x data mixes (backtest months: {months}) ===")
    for w in weight_options:
        mix = "home_only" if w is None else f"home+external(w={w})"
        for name, model in CANDIDATE_MODELS.items():
            t0 = time.time()
            try:
                bt = backtest(model, df, w, months)
            except Exception as e:
                logger.warning(f"  {name:24s} {mix:26s} FAILED ({e})")
                continue
            for rule in (False, True):
                acc, per_month, _, _ = bt[rule]
                results.append({"model": name, "external_weight": w, "mix": mix, "explicit_rule": rule,
                                "backtest_accuracy": acc, "per_month": per_month})
            logger.info(f"  {name:24s} {mix:26s} backtest={bt[False][0]:.3f}  "
                        f"+explicit-rule={bt[True][0]:.3f}  "
                        f"{ {k: round(v, 3) for k, v in bt[True][1].items()} }  ({time.time() - t0:.0f}s)")

    # Ties go to the simpler option (rule off, home-only).
    def key(r):
        return (round(r["backtest_accuracy"], 4), -int(r["explicit_rule"]))
    best_home = max([r for r in results if r["external_weight"] is None], key=key)
    ext = [r for r in results if r["external_weight"] is not None]
    best = best_home
    if ext:
        best_ext = max(ext, key=key)
        gain = best_ext["backtest_accuracy"] - best_home["backtest_accuracy"]
        logger.info(f"Best home-only: {best_home['backtest_accuracy']:.3f}; best with external data: "
                    f"{best_ext['backtest_accuracy']:.3f} (gain {gain * 100:+.1f} pts, "
                    f"need >= {MIN_GAIN_FOR_EXTERNAL * 100:.0f})")
        if gain >= MIN_GAIN_FOR_EXTERNAL:
            best = best_ext
    logger.info(f"Winner by pooled backtest: {best['model']} on {best['mix']}, "
                f"explicit-mention rule={'on' if best['explicit_rule'] else 'off'} "
                f"({best['backtest_accuracy']:.3f})")
    return best, results


def report_area_breakdown(df):
    """Device x area cross-tab by total delay minutes."""
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
    home = df[df["is_home"]]
    logger.info(f"{len(df)} labeled events ({len(home)} home-site, {len(df) - len(home)} external) "
                f"across {df['primary_device'].nunique()} classes")

    report_area_breakdown(home)

    months = get_backtest_months(home)
    best, all_results = compare_all(df, months)
    winner_name, ext_weight, rule = best["model"], best["external_weight"], best["explicit_rule"]
    winner_model = CANDIDATE_MODELS[winner_name]

    # Reports + confusion matrix from the backtest predictions.
    acc, per_month, y_true, y_pred = backtest(winner_model, df, ext_weight, months)[rule]
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(config.HOLDOUT_REPORT_PATH)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix (backtest {months[0]}..{months[-1]}) - {winner_name}")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    logger.info("Computing home-site CV for reference...")
    cv_acc = home_cv(winner_model, df, ext_weight, rule)

    # Final model: every home month + external rows (if the winning mix uses them).
    logger.info(f"Training final '{winner_name}' model on full data...")
    train_df, w = training_subset(df, ext_weight)
    X, vectorizer, known_areas = prepare_modeling_data(train_df, fit=True)
    y = train_df["primary_device"]
    final_model = _clone(winner_model)
    final_model.fit(X, y, sample_weight=w)
    pd.DataFrame(report).transpose().to_csv(config.CLASSIFICATION_REPORT_PATH)

    feature_names = X.columns.tolist()
    n_model_features = getattr(final_model, "n_features_in_", len(feature_names))
    if n_model_features != len(feature_names):
        raise RuntimeError(
            f"Refusing to save: model trained on {n_model_features} features but "
            f"feature_names has {len(feature_names)} entries."
        )

    home_rows = train_df[train_df["is_home"]]
    home_classes = sorted(home_rows["primary_device"].unique().tolist())
    named = home_rows["reason_text"].map(explicit_device_mention)
    fired = named.notna() & named.isin(home_classes)
    rule_precision = float((named[fired] == home_rows.loc[fired, "primary_device"]).mean()) if fired.any() else None
    joblib.dump(final_model, config.MODEL_PATH, compress=3)
    joblib.dump(vectorizer, config.VECTORIZER_PATH, compress=3)
    joblib.dump({"feature_names": feature_names, "known_areas": known_areas,
                 "home_classes": home_classes, "explicit_mention_rule": bool(rule),
                 "explicit_rule_precision": rule_precision}, config.FEATURE_NAMES_PATH)

    latest = months[-1] if months else None
    manifest = {
        "trained_at": datetime.now().isoformat(),
        "winner_model": winner_name,
        "training_data_mix": best["mix"],
        "external_sample_weight": ext_weight,
        "explicit_mention_rule": bool(rule),
        "explicit_rule_coverage": round(float(fired.mean()), 4),
        "explicit_rule_precision": round(rule_precision, 4) if rule_precision is not None else None,
        "min_gain_required_for_external_data": MIN_GAIN_FOR_EXTERNAL,
        "n_features": len(feature_names),
        "n_samples": len(X),
        "n_home_samples": int(train_df["is_home"].sum()),
        "n_external_samples": int((~train_df["is_home"]).sum()),
        "n_classes": int(y.nunique()),
        "classes": sorted(y.unique().tolist()),
        "home_classes": home_classes,
        "evaluation": "rolling-origin backtest on home-site months (train strictly before, test on unseen month)",
        "backtest_months": months,
        "backtest_n_test_events": len(y_true),
        "backtest_accuracy": round(acc, 4),
        "backtest_per_month": {k: round(v, 4) for k, v in per_month.items()},
        "holdout_month": latest,
        "holdout_accuracy": round(per_month.get(latest), 4) if latest in per_month else None,
        "cv_accuracy": round(cv_acc, 4),
        "all_candidates": [
            {"model": r["model"], "mix": r["mix"], "explicit_rule": r["explicit_rule"],
             "backtest_accuracy": round(r["backtest_accuracy"], 4),
             "per_month": {k: round(v, 4) for k, v in r["per_month"].items()}}
            for r in sorted(all_results, key=lambda r: -r["backtest_accuracy"])
        ],
    }
    with open(config.MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"FINAL: winner={winner_name} mix={best['mix']} rule={rule} backtest={acc:.3f} "
                f"latest-month={manifest['holdout_accuracy']} CV(home)={cv_acc:.3f}")
    return final_model


if __name__ == "__main__":
    train_and_evaluate()
