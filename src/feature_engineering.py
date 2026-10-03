"""
Feature engineering -- the ONE implementation used by training,
prediction, and every test. Never reimplemented elsewhere.

Includes area as a real feature this time (previous versions of this
project only used text), since area-wise breakdown is now a stated
requirement.

Multi-site: also includes a one-hot plant_type feature (steel rolling
mill vs. nuclear power plant ...), so the model can learn that some
device classes only occur at some kinds of plant while still sharing
device vocabulary ("pressure switch", "flow switch", "LVDT") across
all of them. The plant type is always known at prediction time (it's
the site the delay is logged at), so this is not leakage.

Text features: word 1-2 grams PLUS character 3-5 grams. The char
n-grams were added because the delay log is full of misspellings and
shorthand that word n-grams treat as unrelated tokens -- confirmed in
the real data: "continous", "missisng", "luberication", "proxy" (for
proximity), "fce" (furnace).
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

# Fixed list (not learned from data) -> the plant-type one-hot columns
# are always identical between training and inference.
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
    """Word + char n-gram TF-IDF. Vocabulary caps scale with data size
    -- the old fixed 200-feature cap was sized for ~350 rows and threw
    away most of the vocabulary once more data was available."""
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
    # FIX: 'mins' (delay duration) was silently included as a numeric
    # feature -- confirmed this causes a real crash (Logistic Regression
    # rejects NaN, and one row has a missing duration) and is also
    # conceptually wrong: duration is an OUTCOME of the delay, not a
    # signal that should predict which device caused it, and it isn't
    # reliably known before the event is fully resolved anyway.
    X = X.drop(columns=["mins"], errors="ignore")
    # Whitelist by prefix: external rows carry extra numeric metadata
    # (power_lost_pct, n_device_classes_mentioned, ...) that must never
    # become features -- they are outcomes or labeling artifacts, the
    # same class of leak as 'mins' above.
    keep = [c for c in X.columns
            if c.startswith(FEATURE_PREFIXES) or c in ("text_length", "word_count")]
    return X[keep], vectorizer, known_areas


FEATURE_PREFIXES = ("kw_", "area_", "pt_", "tfidf_")


def align_features(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Reindex to the exact training-time feature set/order, filling
    any missing column with 0. Call this right before model.predict at
    inference time."""
    return X.reindex(columns=feature_names, fill_value=0)