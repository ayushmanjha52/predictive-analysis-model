"""
Fleet-wide insights -- what the wider power-generation fleet (public
NRC event reports across every US power reactor + the WRI Global Power
Plant Database) says about the same field devices the Combi Mill is
trying to make more reliable.

Every number here is computed from real public records (see
external_data.py for sources and the labeling method). Device labels on
fleet events are keyword-derived (weak labels) -- reported as such in
every response via `label_method`, never presented as hand-tagged.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import re
from functools import lru_cache

import pandas as pd

import config
from data_loader import load_external_events, load_master_events

logger = logging.getLogger(__name__)

LABEL_METHOD = ("Fleet device labels are keyword-derived from NRC event narratives (weak labels; "
                "events naming 2+ device classes are left unlabeled).")

# Evidence-based actions per failure mode -- what the fleet narratives
# say went wrong maps to what to check first at the Combi Mill.
FAILURE_MODE_ACTIONS = {
    "CALIBRATION_DRIFT": "add setpoint/as-found checks to the PM schedule and trend drift between calibrations",
    "WIRING_CONNECTION": "inspect terminations, connectors and cable routing near heat/vibration sources",
    "MOISTURE_CORROSION": "seal enclosures and cable glands; check for water ingress from cooling/descaling sprays",
    "MECHANICAL_VIBRATION": "check mounting brackets and linkages; add vibration isolation where the sensor sits on moving equipment",
    "POWER_SUPPLY": "monitor sensor supply voltage/fuses; add supply-healthy alarms",
    "COMPONENT_AGING": "track device age and replace proactively at end of rated life instead of run-to-failure",
    "MAINTENANCE_HUMAN_ERROR": "use post-maintenance functional tests and checklists after any work near the device",
    "CONTAMINATION_BLOCKAGE": "increase cleaning/purging frequency and protect sensing faces/ports from scale and dust",
}


@lru_cache(maxsize=1)
def _external():
    return load_external_events()


def clear_cache():
    _external.cache_clear()
    _wri.cache_clear()


def _labeled(ext):
    return ext[ext["primary_device"].notna()]


def _split_tags(series):
    return series.dropna().astype(str).str.split("|").explode().loc[lambda s: s != ""]


def fleet_overview() -> dict:
    ext = _external()
    if len(ext) == 0:
        return {"available": False, "message": "No external fleet data. Run `python src/external_data.py`."}
    lab = _labeled(ext)
    dates = pd.to_datetime(ext["date"], errors="coerce")
    return {
        "available": True,
        "source": "US NRC Event Notification Reports (power reactors)",
        "total_events": int(len(ext)),
        "field_device_events": int(len(lab)),
        "plants": int(ext["site"].nunique()),
        "states": int(ext["state"].nunique()),
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
        "reactor_trips": int(ext["reactor_trip"].sum()),
        "grid_related_events": int(ext["grid_related"].sum()),
        "field_device_trips": int(lab["reactor_trip"].sum()),
        "label_method": LABEL_METHOD,
    }


def device_ranking() -> dict:
    """Per device class across the fleet: how often it fails, how many
    plants, how often a failure trips the reactor (objective severity),
    plus the Combi Mill's own numbers for the same device."""
    ext = _external()
    lab = _labeled(ext)
    if len(lab) == 0:
        return {"devices": [], "label_method": LABEL_METHOD}

    g = lab.groupby("primary_device").agg(
        fleet_events=("site", "size"),
        plants_affected=("site", "nunique"),
        reactor_trips=("reactor_trip", "sum"),
        grid_related=("grid_related", "sum"),
        avg_power_lost_pct=("power_lost_pct", lambda s: s[s > 0].mean()),
    ).reset_index().rename(columns={"primary_device": "device"})
    g["trip_rate_pct"] = (g["reactor_trips"] / g["fleet_events"] * 100).round(1)
    g["fleet_share_pct"] = (g["fleet_events"] / g["fleet_events"].sum() * 100).round(1)
    g["avg_power_lost_pct"] = g["avg_power_lost_pct"].round(1)

    home = load_master_events()
    home = home[home["primary_device"].notna()]
    h = home.groupby("primary_device").agg(home_events=("mins", "size"),
                                           home_minutes=("mins", "sum")).reset_index() \
        .rename(columns={"primary_device": "device"})
    g = g.merge(h, on="device", how="outer")
    for col in ("fleet_events", "home_events", "plants_affected", "reactor_trips", "grid_related"):
        g[col] = g[col].fillna(0).astype(int)
    g["low_sample"] = g["fleet_events"] < config.MIN_CLASS_COUNT
    g["home_minutes"] = g["home_minutes"].fillna(0).round(0)
    g = g.sort_values(["fleet_events", "home_events"], ascending=False)
    g = g.astype(object).where(g.notna(), None)
    return {"devices": g.to_dict(orient="records"), "label_method": LABEL_METHOD}


def failure_modes(device: str = None) -> dict:
    """What actually went wrong, mined from fleet narratives, per device."""
    lab = _labeled(_external())
    if device:
        lab = lab[lab["primary_device"] == device.upper()]
    out = []
    for dev, grp in lab.groupby("primary_device"):
        tags = _split_tags(grp["failure_modes"])
        n = len(grp)
        modes = tags.value_counts()
        out.append({
            "device": dev,
            "events": int(n),
            "modes": [{"mode": m, "events": int(c), "pct": round(c / n * 100, 1)} for m, c in modes.items()],
        })
    out.sort(key=lambda r: -r["events"])
    return {"by_device": out, "label_method": LABEL_METHOD}


def cross_site_lessons() -> dict:
    """For each device the Combi Mill tracks that ALSO appears in the
    power-plant fleet: how it ranks in both places, how severe fleet
    failures are, the top real failure modes, and a concrete action."""
    ranking = {r["device"]: r for r in device_ranking()["devices"]}
    modes = {r["device"]: r for r in failure_modes()["by_device"]}
    lessons = []
    for dev, r in ranking.items():
        if not r["home_events"] or not r["fleet_events"]:
            continue
        top_modes = (modes.get(dev, {}).get("modes") or [])[:3]
        actions = [FAILURE_MODE_ACTIONS[m["mode"]] for m in top_modes if m["mode"] in FAILURE_MODE_ACTIONS][:2]
        lessons.append({
            "device": dev,
            "home_events": r["home_events"],
            "home_minutes": r["home_minutes"],
            "fleet_events": r["fleet_events"],
            "fleet_plants_affected": r["plants_affected"],
            "fleet_trip_rate_pct": r["trip_rate_pct"],
            "top_failure_modes": top_modes,
            "recommended_actions": actions,
        })
    lessons.sort(key=lambda r: -(r["home_minutes"] or 0))
    return {"lessons": lessons, "label_method": LABEL_METHOD}


def yearly_trend(top_n_devices: int = 6) -> dict:
    ext = _external().copy()
    ext["year"] = pd.to_datetime(ext["date"], errors="coerce").dt.year
    ext = ext[ext["year"].notna()]
    ext["year"] = ext["year"].astype(int)
    years = sorted(ext["year"].unique().tolist())
    totals = ext.groupby("year").agg(events=("site", "size"), trips=("reactor_trip", "sum"),
                                     grid=("grid_related", "sum")).reindex(years, fill_value=0)
    lab = _labeled(ext)
    top = lab["primary_device"].value_counts().head(top_n_devices).index.tolist()
    per_dev = lab[lab["primary_device"].isin(top)].groupby(["primary_device", "year"]).size()
    return {
        "years": years,
        "total_events": totals["events"].astype(int).tolist(),
        "reactor_trips": totals["trips"].astype(int).tolist(),
        "grid_events": totals["grid"].astype(int).tolist(),
        "devices": {d: [int(per_dev.get((d, y), 0)) for y in years] for d in top},
        "note": "The latest year is partial (data up to the most recent NRC report).",
    }


def grid_insights(top_n: int = 10) -> dict:
    """Grid / offsite-power events -- the transmission-grid side of
    plant reliability: how often, what caused them, how often they
    tripped the unit, and which plants see them most."""
    ext = _external()
    grid = ext[ext["grid_related"]].copy()
    if len(grid) == 0:
        return {"events": 0}
    grid["year"] = pd.to_datetime(grid["date"], errors="coerce").dt.year
    causes = _split_tags(grid["grid_causes"]).value_counts()
    sites = grid.groupby("site").agg(events=("site", "size"), trips=("reactor_trip", "sum"),
                                     state=("state", "first")).reset_index() \
        .sort_values("events", ascending=False).head(top_n)
    by_year = grid.groupby("year").size()
    return {
        "events": int(len(grid)),
        "trip_rate_pct": round(grid["reactor_trip"].mean() * 100, 1),
        "plants_affected": int(grid["site"].nunique()),
        "causes": [{"cause": c, "events": int(n), "pct": round(n / len(grid) * 100, 1)} for c, n in causes.items()],
        "uncategorized_pct": round(grid["grid_causes"].fillna("").eq("").mean() * 100, 1),
        "by_year": [{"year": int(y), "events": int(n)} for y, n in by_year.items() if pd.notna(y)],
        "top_plants": sites.to_dict(orient="records"),
    }


# ------------------------------------------------------------------ WRI --
@lru_cache(maxsize=1)
def _wri():
    if not config.WRI_PLANTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(config.WRI_PLANTS_PATH, low_memory=False)


def _name_key(name):
    """'Vogtle 3/4' / 'Vogtle Electric Generating Plant' -> 'vogtle'."""
    words = re.findall(r"[a-z]+", str(name).lower())
    stop = {"nuclear", "generating", "station", "plant", "power", "electric", "energy", "center", "unit", "units"}
    words = [w for w in words if w not in stop]
    return words[0] if words else ""


MIN_EVENTS_FOR_PLANT_RANKING = 10


def plant_reliability(top_n: int = 15) -> dict:
    """Unit trips (reactor/turbine) per GW of installed capacity, per
    NRC plant -- normalizes raw counts so a big multi-unit site isn't
    ranked worst just for being big. Uses ALL reported events (trips are
    an objective outcome, not a keyword label), and only ranks plants
    with >= MIN_EVENTS_FOR_PLANT_RANKING events. Capacity comes from the
    WRI database, matched by plant name (unmatched plants are reported,
    not guessed)."""
    ext = _external()
    wri = _wri()
    if len(ext) == 0 or len(wri) == 0:
        return {"plants": [], "message": "Needs both NRC and WRI data."}
    us_nuc = wri[(wri["country"] == "USA") & (wri["primary_fuel"] == "Nuclear")].copy()
    us_nuc["key"] = us_nuc["name"].map(_name_key)
    cap = us_nuc.groupby("key")["capacity_mw"].sum()

    g = ext.groupby("site").agg(events=("site", "size"), trips=("reactor_trip", "sum"),
                                grid_events=("grid_related", "sum"),
                                device_events=("primary_device", "count"),
                                state=("state", "first")).reset_index()
    g["key"] = g["site"].map(_name_key)
    g["capacity_mw"] = g["key"].map(cap)
    matched = g[g["capacity_mw"].notna() & (g["events"] >= MIN_EVENTS_FOR_PLANT_RANKING)].copy()
    gw = matched["capacity_mw"] / 1000
    matched["trips_per_gw"] = (matched["trips"] / gw).round(1)
    matched["events_per_gw"] = (matched["events"] / gw).round(1)
    matched = matched.sort_values("trips_per_gw", ascending=False).head(top_n)
    return {
        "plants": matched.drop(columns=["key"]).to_dict(orient="records"),
        "matched_plants": int(g["capacity_mw"].notna().sum()),
        "unmatched_plants": int(g["capacity_mw"].isna().sum()),
        "min_events": MIN_EVENTS_FOR_PLANT_RANKING,
    }


def power_plants_summary(country: str = None, top_n: int = 10) -> dict:
    """Global installed base (WRI Global Power Plant Database)."""
    wri = _wri()
    if len(wri) == 0:
        return {"available": False, "message": "WRI data not downloaded. Run `python src/external_data.py`."}
    scope = wri
    if country:
        scope = wri[wri["country"].str.upper() == country.upper()]
        if len(scope) == 0:
            return {"available": False, "message": f"No plants found for country '{country}'."}
    by_fuel = scope.groupby("primary_fuel").agg(plants=("name", "size"), capacity_mw=("capacity_mw", "sum")) \
        .reset_index().sort_values("capacity_mw", ascending=False)
    by_fuel["capacity_gw"] = (by_fuel["capacity_mw"] / 1000).round(1)
    by_fuel["share_pct"] = (by_fuel["capacity_mw"] / by_fuel["capacity_mw"].sum() * 100).round(1)
    result = {
        "available": True,
        "source": "WRI Global Power Plant Database (CC BY 4.0)",
        "scope": country.upper() if country else "WORLD",
        "plants": int(len(scope)),
        "capacity_gw": round(scope["capacity_mw"].sum() / 1000, 1),
        "by_fuel": by_fuel.drop(columns=["capacity_mw"]).to_dict(orient="records"),
        "largest_plants": scope.nlargest(top_n, "capacity_mw")[
            ["name", "country_long", "primary_fuel", "capacity_mw"]].to_dict(orient="records"),
    }
    if not country:
        top_c = wri.groupby(["country", "country_long"])["capacity_mw"].sum().reset_index() \
            .nlargest(top_n, "capacity_mw")
        top_c["capacity_gw"] = (top_c["capacity_mw"] / 1000).round(1)
        result["top_countries"] = top_c.drop(columns=["capacity_mw"]).to_dict(orient="records")
    return result


def sites_summary() -> dict:
    home = load_master_events()
    ext = _external()
    sites = [{"site": config.HOME_SITE, "plant_type": config.HOME_PLANT_TYPE, "events": int(len(home)),
              "field_device_events": int(home["primary_device"].notna().sum()), "is_home": True}]
    if len(ext):
        g = ext.groupby(["site", "plant_type"]).agg(events=("site", "size"),
                                                   field_device_events=("primary_device", "count")) \
            .reset_index().sort_values("events", ascending=False)
        g["is_home"] = False
        sites += g.to_dict(orient="records")
    by_type = {}
    for s in sites:
        t = by_type.setdefault(s["plant_type"], {"plant_type": s["plant_type"], "sites": 0, "events": 0})
        t["sites"] += 1
        t["events"] += s["events"]
    return {"by_plant_type": list(by_type.values()), "sites": sites}
