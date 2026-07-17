"""
FMEA risk lookup + fusion with empirical Pareto data.

Devices with no FMEA sheet get risk_level="UNKNOWN" + is_estimated=True
-- NEVER a fabricated "Low". Confirmed important in this project:
Photocell has repeatedly been a top empirical delay contributor while
sometimes lacking an FMEA sheet -- silently defaulting it to "Low"
would be actively misleading.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import pandas as pd

import config

logger = logging.getLogger(__name__)


def _load_fmea_data():
    if not config.FMEA_SCORES_PATH.exists():
        logger.warning(f"FMEA file not found at {config.FMEA_SCORES_PATH} -- "
                        f"all devices will show UNKNOWN risk until this is added.")
        return None
    try:
        df = pd.read_csv(config.FMEA_SCORES_PATH)
        if "device" not in df.columns:
            logger.error(f"{config.FMEA_SCORES_PATH} missing 'device' column.")
            return None
        df["device"] = df["device"].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        logger.error(f"Failed to load FMEA scores: {e}")
        return None


def load_fmea_scores() -> pd.DataFrame:
    df = _load_fmea_data()
    if df is None:
        return pd.DataFrame(columns=["device", "rpn", "severity", "occurrence", "detection", "risk_level"])
    return df


def get_fmea_risk(device: str) -> dict:
    df = _load_fmea_data()
    device_key = device.upper().strip()
    if df is not None:
        row = df[df["device"] == device_key]
        if len(row):
            r = row.iloc[0]
            return {
                "rpn": int(r["rpn"]), "severity": float(r["severity"]),
                "occurrence": float(r["occurrence"]), "detection": float(r["detection"]),
                "risk_level": r["risk_level"], "is_estimated": False,
            }
    return {"rpn": None, "severity": None, "occurrence": None, "detection": None,
            "risk_level": "UNKNOWN", "is_estimated": True}


def fuse_pareto_with_fmea(pareto: pd.DataFrame, fmea: pd.DataFrame) -> pd.DataFrame:
    fused = pareto.merge(fmea, on="device", how="left")
    fused["is_estimated"] = fused["rpn"].isna()
    fused["risk_level"] = fused["risk_level"].fillna("UNKNOWN")
    for col in ("rpn", "severity", "occurrence", "detection"):
        fused[col] = fused[col].astype(object).where(fused[col].notna(), None)
    return fused


RECOMMENDATIONS = {
    "PHOTOCELL": "Improve air/dust purging and increase lens cleaning frequency; "
                 "high recurrence with moderate severity suggests a process-discipline "
                 "fix (cleaning cadence) rather than a redesign.",
    "LVDT": "Investigate core/transducer sticking and calibration drift; high FMEA "
            "severity combined with high avg delay duration makes this a priority "
            "for mechanical inspection, not just software recalibration.",
    "ENCODER": "Check coupling/mounting integrity and cable routing; encoder failures "
               "tend to be high-severity per incident -- prioritize redundant "
               "sensing or improved mounting on the highest-volume stands.",
    "HMD": "Verify sensitivity calibration and air purging on the optical path; "
           "frequent but lower-severity -- a maintenance checklist fix is likely "
           "sufficient before considering hardware changes.",
    "PROXIMITY": "Inspect sensor gap/target alignment; recurring issues often trace "
                 "to mechanical misalignment after maintenance work, not sensor failure itself.",
    "PRESSURE_SWITCH": "Check hydraulic supply stability and switch calibration; "
                       "high average duration per incident suggests these take longer "
                       "to diagnose -- consider a dedicated troubleshooting checklist.",
    "FLOW_SWITCH": "Verify coolant/water flow consistency and switch fouling; "
                   "typically a maintenance-frequency fix.",
    "LASER": "Check head alignment after any crane or mechanical work nearby; "
             "low volume but worth monitoring for trend changes.",
}


def get_recommendation(device: str) -> str:
    return RECOMMENDATIONS.get(device.upper(), "No specific recommendation on file for this device yet.")


if __name__ == "__main__":
    for d in ["HMD", "LVDT", "ENCODER", "PHOTOCELL", "PROXIMITY"]:
        print(f"{d:16s} -> {get_fmea_risk(d)}")