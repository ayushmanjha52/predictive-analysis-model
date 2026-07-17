"""
Feature engineering -- the ONE implementation used by training,
prediction, and every test. Never reimplemented elsewhere.

Includes area as a real feature this time (previous versions of this
project only used text), since area-wise breakdown is now a stated
requirement.
"""
import re
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DOMAIN_KEYWORDS = {
    "kw_cleaning": ["clean", "dust", "dirty", "contamina"],
    "kw_alignment": ["alignment", "misalign", "position error", "positioning"],
    "kw_electrical": ["voltage", "power", "cable", "psu", "signal"],
    "kw_mechanical": ["bolt", "vibration", "loose", "coupling", "bracket"],
    "kw_pressure": ["pressure", "oil", "lubrica", "hydraulic"],
    "kw_missing_feedback": ["feedback missing", "not sensing", "no signal", "missed detection"],
    "kw_low_high": ["low", "high", "not ok", "malfunction"],
    "kw_cobble": ["cobble", "autochopped", "chopped"],
    "kw_hot_out": ["hot out", "hotout"],
    "kw_overtravel": ["overtravel", "over travelled"],
}


def clean_text(text) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def create_domain_features(df: pd.DataFrame, text_col: str = "clean_text") -> pd.DataFrame:
    """Binary keyword-group flags + text stats. Every column computed
    from the row's own text -- safe to call on a single new row at
    inference time, no cross-row statistics involved."""
    df = df.copy()
    texts = df[text_col].fillna("")
    for feat_name, keywords in DOMAIN_KEYWORDS.items():
        df[feat_name] = texts.apply(lambda t: int(any(kw in t for kw in keywords)))
    df["text_length"] = texts.apply(len)
    df["word_count"] = texts.apply(lambda t: len(t.split()))
    return df


def create_area_features(df: pd.DataFrame, area_col: str = "area", known_areas=None) -> pd.DataFrame:
    """
    One-hot area features. `known_areas` MUST be passed at inference
    time (the exact list from training) so a new row produces the same
    columns as training did, in the same set -- this is the same
    train/predict alignment principle as the TF-IDF vectorizer, applied
    to area instead of text.
    """
    df = df.copy()
    area_clean = df[area_col].fillna("UNKNOWN").astype(str).str.upper().str.strip()
    if known_areas is None:
        known_areas = sorted(area_clean.unique())
    for a in known_areas:
        col = f"area_{a}".replace(" ", "_").replace("-", "_")
        df[col] = (area_clean == a).astype(int)
    return df, known_areas


def build_tfidf_features(texts, vectorizer, fit: bool):
    if fit:
        matrix = vectorizer.fit_transform(texts)
    else:
        matrix = vectorizer.transform(texts)
    cols = [f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]
    return pd.DataFrame(matrix.toarray(), columns=cols, index=texts.index)


def prepare_modeling_data(df: pd.DataFrame, vectorizer: TfidfVectorizer = None,
                            fit: bool = True, known_areas=None):
    """
    End-to-end: raw reason_text + area -> full feature DataFrame.

    fit=True  (training): fits a NEW vectorizer, derives known_areas
              from this data. Returns (X, vectorizer, known_areas).
    fit=False (inference/eval): reuses the vectorizer AND known_areas
              passed in -- raises if either is missing, rather than
              silently fitting fresh ones (that silent-refit was the
              root cause of an earlier production bug in this project).
    """
    df = df.copy()
    df["clean_text"] = df["reason_text"].apply(clean_text)

    if fit:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200, stop_words="english")
    elif vectorizer is None:
        raise ValueError("fit=False requires a trained vectorizer to be passed in.")

    tfidf_df = build_tfidf_features(df["clean_text"], vectorizer, fit=fit)
    tfidf_df.index = df.index

    df = create_domain_features(df, text_col="clean_text")

    if not fit and known_areas is None:
        raise ValueError("fit=False requires known_areas from training to be passed in.")
    df, known_areas = create_area_features(df, known_areas=known_areas)

    combined = pd.concat([df, tfidf_df], axis=1)
    import numpy as np
    X = combined.select_dtypes(include=[np.number])
    # FIX: 'mins' (delay duration) was silently included as a numeric
    # feature -- confirmed this causes a real crash (Logistic Regression
    # rejects NaN, and one row has a missing duration) and is also
    # conceptually wrong: duration is an OUTCOME of the delay, not a
    # signal that should predict which device caused it, and it isn't
    # reliably known before the event is fully resolved anyway.
    X = X.drop(columns=["mins"], errors="ignore")
    return X, vectorizer, known_areas


def align_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Reindex to the exact training-time feature set/order, filling
    any missing column with 0. Call this right before model.predict at
    inference time."""
    return X.reindex(columns=feature_names, fill_value=0)