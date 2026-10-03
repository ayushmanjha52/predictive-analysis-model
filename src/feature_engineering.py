"""
Feature engineering shared by training, prediction and tests.

Features:
  - word 1-2 gram and character 3-5 gram TF-IDF (character n-grams cope
    with the log's misspellings and shorthand: "continous", "missisng",
    "proxy", "fce"),
  - keyword-group flags and text-length stats,
  - one-hot area,
  - one-hot plant type (steel rolling mill, nuclear power, ...). The
    plant type is known at prediction time, so this is not leakage.
"""
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

import config

DOMAIN_KEYWORDS = {
    "kw_cleaning": ["clean", "dust", "dirty", "contamina"],
    "kw_alignment": ["alignment", "misalign", "position error", "positioning"],
    "kw_electrical": ["voltage", "power", "cable", "psu", "signal"],
    "kw_mechanical": ["bolt", "vibration", "loose", "coupling", "bracket"],
    "kw_pressure": ["pressure", "oil", "lubrica", "luberica", "hydraulic"],
    "kw_missing_feedback": ["feedback missing", "not sensing", "no signal", "missed detection"],
    "kw_low_high": ["low", "high", "not ok", "malfunction"],
    "kw_cobble": ["cobble", "autochopped", "chopped"],
    "kw_hot_out": ["hot out", "hotout"],
    "kw_overtravel": ["overtravel", "over travelled"],
    "kw_flow": ["flow", "water", "coolant", "cooling water"],
    "kw_level": ["level", "tank", "float"],
    "kw_temperature": ["temperature", "temp ", "overtemp", "thermocouple", "rtd"],
    "kw_feedback_position": ["feedback", "position", "stroke", "travel"],
    "kw_detection": ["sensing", "detect", "flicker", "continous", "continuous"],
    "kw_counting": ["encoder", "count", "pulse", "speed", "length"],
    "kw_trip": ["trip", "scram", "shutdown", "actuation"],
}

FEATURE_PREFIXES = ("kw_", "area_", "pt_", "tfidf_")

# Fixed list so plant-type columns are identical at train and inference time.
PLANT_TYPES = [config.HOME_PLANT_TYPE, "NUCLEAR_POWER", "THERMAL_POWER", "HYDRO_POWER",
               "GRID_SUBSTATION", "OTHER"]


def clean_text(text) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def create_domain_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Binary keyword-group flags + text stats (per-row, no cross-row statistics)."""
    df = df.copy()
    texts = df[text_col].fillna("")
    for feat_name, keywords in DOMAIN_KEYWORDS.items():
        df[feat_name] = texts.apply(lambda t: int(any(kw in t for kw in keywords)))
    df["text_length"] = texts.apply(len)
    df["word_count"] = texts.apply(lambda t: len(t.split()))
    return df


def create_area_features(df: pd.DataFrame, area_col: str = "area", known_areas=None) -> pd.DataFrame:
    """One-hot area features. Pass the training-time `known_areas` at
    inference so the columns match."""
    df = df.copy()
    area_clean = df[area_col].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    if known_areas is None:
        known_areas = sorted(area_clean.unique())
    for a in known_areas:
        col = f"area_{a}".replace(" ", "_").replace("-", "_")
        df[col] = (area_clean == a).astype(int)
    return df, known_areas


def create_plant_type_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "plant_type" in df.columns:
        pt = df["plant_type"].fillna(config.HOME_PLANT_TYPE).astype(str).str.upper()
    else:
        pt = pd.Series(config.HOME_PLANT_TYPE, index=df.index)
    pt = pt.where(pt.isin(PLANT_TYPES), "OTHER")
    for p in PLANT_TYPES:
        df[f"pt_{p}"] = (pt == p).astype(int)
    return df


def make_vectorizer(n_docs: int):
    """Word + char n-gram TF-IDF; vocabulary caps scale with data size."""
    word_cap = 300 if n_docs < 1000 else 1500
    char_cap = 300 if n_docs < 1000 else 1500
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=word_cap,
                                 stop_words="english", sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=char_cap, sublinear_tf=True)),
    ])


def build_tfidf_features(texts, vectorizer, fit: bool):
    if fit:
        matrix = vectorizer.fit_transform(texts)
    else:
        matrix = vectorizer.transform(texts)
    cols = [f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]
    return pd.DataFrame(matrix.toarray().astype(np.float32), columns=cols, index=texts.index)


def prepare_modeling_data(df: pd.DataFrame, vectorizer: TfidfVectorizer = None,
                            fit: bool = True, known_areas=None):
    """
    reason_text + area (+ plant_type) -> feature DataFrame.

    fit=True:  fits a new vectorizer and derives known_areas.
    fit=False: requires the training-time vectorizer and known_areas.
    Returns (X, vectorizer, known_areas).
    """
    df = df.copy()
    df["clean_text"] = df["reason_text"].apply(clean_text)

    if fit:
        vectorizer = make_vectorizer(len(df))
    elif vectorizer is None:
        raise ValueError("fit=False requires a trained vectorizer to be passed in.")

    tfidf_df = build_tfidf_features(df["clean_text"], vectorizer, fit=fit)
    tfidf_df.index = df.index

    df = create_domain_features(df, text_col="clean_text")

    if not fit and known_areas is None:
        raise ValueError("fit=False requires known_areas from training to be passed in.")
    df, known_areas = create_area_features(df, known_areas=known_areas)
    df = create_plant_type_features(df)

    combined = pd.concat([df, tfidf_df], axis=1)
    X = combined.select_dtypes(include=[np.number])
    # Whitelist feature columns: other numeric columns (delay minutes,
    # power lost, labeling metadata) are outcomes, not predictors.
    keep = [c for c in X.columns
            if c.startswith(FEATURE_PREFIXES) or c in ("text_length", "word_count")]
    return X[keep], vectorizer, known_areas


def align_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Reindex to the training-time feature set/order (missing columns -> 0)."""
    return X.reindex(columns=feature_names, fill_value=0)