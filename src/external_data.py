"""
Public power-sector data.

1. NRC Event Notification Reports (US Nuclear Regulatory Commission)
   https://www.nrc.gov/reading-rm/doc-collections/event-status/event/
   Reportable events at US power reactors, each with a plant-written
   narrative, SCRAM code and power levels. Only "Power Reactor" events
   are kept.

2. WRI Global Power Plant Database (CC BY 4.0)
   https://github.com/wri/global-power-plant-database
   About 35,000 plants: capacity, fuel, country, location.

Device labels: NRC narratives have no device field, so an event is
labeled when exactly one device class is named and failure language
appears near the mention. The training text is the sentence(s) around
the mention. These rows are never used to measure accuracy.

Run:  python src/external_data.py            (fetch; resumable)
      python src/external_data.py --relabel  (re-label cached raw data)
"""
import argparse
import datetime as dt
import html
import io
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
import pandas as pd

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NRC_URL = "https://www.nrc.gov/reading-rm/doc-collections/event-status/event/{y}/{ymd}en.html"
WRI_URL = ("https://raw.githubusercontent.com/wri/global-power-plant-database/master/"
           "output_database/global_power_plant_database.csv")
USER_AGENT = "pdm-research/1.0 (field-device reliability study; public data)"

# Device vocabulary. Events naming two or more classes stay unlabeled.
# Class names match the mill's where the device is the same.
DEVICE_PATTERNS = {
    "PRESSURE_SWITCH": r"pressure switch(?:es)?",
    "FLOW_SWITCH": r"flow switch(?:es)?",
    "LEVEL_SWITCH": r"level switch(?:es)?|float switch(?:es)?",
    "TEMPERATURE_SWITCH": r"temperature switch(?:es)?|thermostat",
    "LIMIT_SWITCH": r"limit switch(?:es)?|position switch(?:es)?",
    "LVDT": r"\blvdts?\b|linear variable differential",
    "PROXIMITY": r"proximity (?:probe|switch|sensor)s?|prox(?:imity)? probes?",
    "ENCODER": r"\bencoders?\b",
    "PHOTOCELL": r"photo ?cells?|photoelectric",
    "LASER": r"\blaser (?:sensor|detector|gauge)s?",
    "PRESSURE_TRANSMITTER": r"pressure transmitters?|\bp/?t-?\d",
    "FLOW_TRANSMITTER": r"flow transmitters?",
    "LEVEL_TRANSMITTER": r"level transmitters?",
    "TEMPERATURE_SENSOR": r"\brtds?\b|resistance temperature detector|thermocouples?",
    "SPEED_SENSOR": r"speed (?:sensor|probe|pickup)s?|magnetic pickup",
    "VIBRATION_SENSOR": r"vibration (?:sensor|probe|transducer|monitor)s?|accelerometers?",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in DEVICE_PATTERNS.items()}

# A mention counts as a failure only with failure language nearby.
FAILURE_WORDS = re.compile(
    r"fail|fault|malfunction|spurious|inoperable|degraded|erratic|drift|"
    r"short|open circuit|grounded|broken|stuck|loose|leak|did not|not (?:function|work|respond)|"
    r"tripped|actuat|signal|incorrect|lost|loss of|out of calibration|setpoint",
    re.I,
)

# Grid / offsite-power events.
GRID_PATTERN = re.compile(
    r"loss of (?:all )?off-?site power|grid (?:disturbance|instability|frequency|voltage)|"
    r"switchyard|transmission line|offsite power|off-site power|load reject",
    re.I,
)
TRIP_PATTERN = re.compile(r"reactor trip|automatic (?:reactor )?scram|turbine trip|manual (?:reactor )?scram", re.I)

# Failure modes mined from the full narrative (multi-label).
FAILURE_MODES = {
    "CALIBRATION_DRIFT": r"drift|out of calibration|out of tolerance|calibrat|setpoint|as-found",
    "WIRING_CONNECTION": r"wiring|\bwires?\b|loose connection|terminal|connector|\bcables?\b|open circuit|short(?:ed)? circuit|\bground(?:ed)?\b",
    "MOISTURE_CORROSION": r"moisture|water intrusion|\bwet\b|corrosion|corroded|condensation|humidity",
    "MECHANICAL_VIBRATION": r"vibration|mechanical|bracket|mounting|\bbent\b|linkage|binding|\bstuck\b|\bworn\b",
    "POWER_SUPPLY": r"power supply|\bfuses?\b|blown|loss of power|inverter",
    "COMPONENT_AGING": r"age-related|aging|end of life|degradation|internal failure|component failure",
    "MAINTENANCE_HUMAN_ERROR": r"human error|personnel error|maintenance activit|inadvertent|improperly installed|incorrectly installed",
    "CONTAMINATION_BLOCKAGE": r"debris|\bclog|plugged|blockage|fouling|\bdirt|\bdust|contamina",
}
_FAILURE_MODES = {k: re.compile(v, re.I) for k, v in FAILURE_MODES.items()}

# What caused a grid / offsite-power event.
GRID_CAUSES = {
    "WEATHER": r"storm|hurricane|tornado|lightning|\bice\b|\bsnow|\bwinds?\b|flood|weather|heat wave",
    "SWITCHYARD_EQUIPMENT": r"breaker|transformer|\brelays?\b|insulator|\bbus(?:es)?\b|disconnect switch",
    "GRID_DISTURBANCE": r"grid disturbance|grid instability|grid frequency|voltage (?:dip|transient|fluctuation)|transmission system operator|load dispatcher",
    "HUMAN_ERROR": r"human error|personnel error|inadvertent",
    "FIRE_ANIMAL": r"\bfire\b|animal|\bbirds?\b|squirrel|snake",
}
_GRID_CAUSES = {k: re.compile(v, re.I) for k, v in GRID_CAUSES.items()}


def _tags(text, compiled):
    return "|".join(k for k, rx in compiled.items() if rx.search(text or ""))


BOILERPLATE = re.compile(
    r"the nrc (?:senior )?resident inspector (?:has been|was) notified\.?|"
    r"the following information was provided by the licensee[^.:]*[.:]|"
    r"the licensee (?:has )?notified the [^.]*\.|"
    r"\* \* \* (?:update|retraction)[^*]*\* \* \*",
    re.I,
)

NRC_RAW_PATH = config.EXTERNAL_DIR / "nrc_events_raw.csv.gz"
NRC_DAYS_DONE_PATH = config.EXTERNAL_DIR / "nrc_days_done.txt"
WRI_PATH = config.EXTERNAL_DIR / "global_power_plants.csv.gz"


# ------------------------------------------------------------------ NRC --
class RateLimited(Exception):
    pass


def _fetch(url, retries=3, timeout=40):
    """Page text, None on 404 (no report that day), RateLimited on 403/429."""
    for attempt in range(retries):
        try:
            # nrc.gov rejects requests without an Accept header
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code in (403, 429):
                raise RateLimited(url)
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def _strip_tags(s):
    s = re.sub(r"<br\s*\??/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s).replace("\xa0", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


def _field(block, name):
    m = re.search(rf"<b>{re.escape(name)}:</b>\s*(.*?)(?:<br|</div>|&nbsp;)", block, re.S | re.I)
    return _strip_tags(m.group(1)) if m else None


def parse_nrc_page(page_html):
    """Split one daily report page into individual events. Returns a
    list of dicts (Power Reactor events only)."""
    events = []
    starts = [m.start() for m in re.finditer(r'<div class="grid border" id="en\d+"', page_html)]
    starts.append(len(page_html))
    for a, b in zip(starts, starts[1:]):
        block = page_html[a:b]
        kind = re.search(r'<div class="th">([^<]+)</div>', block)
        if not kind or "power reactor" not in kind.group(1).lower():
            continue
        num = re.search(r"Event Number:\s*(\d+)", block)
        text_m = re.search(r"<b>Event Text</b>\s*<div class=\"border\">(.*?)</div>", block, re.S)
        if not num or not text_m:
            continue

        scram_codes, init_pwr, curr_pwr = [], [], []
        tbl = re.search(r"<tbody>(.*?)</tbody>", block, re.S)
        if tbl:
            for row in re.findall(r"<tr>(.*?)</tr>", tbl.group(1), re.S):
                cells = [_strip_tags(c) for c in re.findall(r"<td>(.*?)</td>", row, re.S)]
                if len(cells) >= 6:
                    scram_codes.append(cells[1])
                    init_pwr.append(pd.to_numeric(cells[3], errors="coerce"))
                    curr_pwr.append(pd.to_numeric(cells[5], errors="coerce"))
        cfr = re.search(r"10 CFR Section:(.*?)</div>", block, re.S)

        events.append({
            "event_number": int(num.group(1)),
            "facility": _field(block, "Facility"),
            "state": _field(block, "State"),
            "region": _field(block, "Region"),
            "rx_type": _field(block, "RX Type"),
            "event_date": _field(block, "Event Date"),
            "notification_date": _field(block, "Notification Date"),
            "emergency_class": _field(block, "Emergency Class"),
            "cfr_section": _strip_tags(cfr.group(1)) if cfr else None,
            "scram": any(c and c.upper() not in ("N", "") for c in scram_codes),
            "scram_codes": ",".join(c for c in scram_codes if c),
            "initial_power": max([p for p in init_pwr if pd.notna(p)], default=None),
            "current_power": min([p for p in curr_pwr if pd.notna(p)], default=None),
            "event_text": _strip_tags(text_m.group(1)),
        })
    return events


def _load_done_days():
    if not NRC_DAYS_DONE_PATH.exists():
        return set()
    return set(NRC_DAYS_DONE_PATH.read_text().split())


def fetch_nrc_events(start: dt.date, end: dt.date, workers: int = 2, delay: float = 0.5,
                     existing: pd.DataFrame = None) -> pd.DataFrame:
    """Resumable: days recorded in nrc_days_done.txt are skipped. On a
    rate-limit response all workers back off exponentially and the day
    is retried."""
    done_days = _load_done_days()
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in days if d.isoformat() not in done_days]
    logger.info(f"NRC: {len(todo)} of {len(days)} days to fetch ({start} .. {end}), "
                f"{workers} workers, {delay}s delay")
    events = [] if existing is None else existing.to_dict("records")
    backoff = {"until": 0.0, "secs": 60}

    def work(day):
        url = NRC_URL.format(y=day.year, ymd=day.strftime("%Y%m%d"))
        while True:
            wait = backoff["until"] - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                page = _fetch(url)
                backoff["secs"] = 60
                time.sleep(delay)
                return day, page
            except RateLimited:
                if backoff["until"] <= time.time():
                    logger.warning(f"  rate-limited by nrc.gov -- pausing {backoff['secs']}s")
                    backoff["until"] = time.time() + backoff["secs"]
                    backoff["secs"] = min(backoff["secs"] * 2, 900)

    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool, open(NRC_DAYS_DONE_PATH, "a") as done_f:
        for fut in as_completed([pool.submit(work, d) for d in todo]):
            day, page = fut.result()
            if page:
                events.extend(parse_nrc_page(page))
            done_f.write(day.isoformat() + "\n")
            done_f.flush()
            n += 1
            if n % 200 == 0:
                logger.info(f"  {n}/{len(todo)} days, {len(events)} power-reactor events total")
                # checkpoint
                pd.DataFrame(events).drop_duplicates("event_number") \
                    .to_csv(NRC_RAW_PATH, index=False, compression="gzip")
    df = pd.DataFrame(events).drop_duplicates("event_number")
    logger.info(f"NRC: {len(df)} unique power-reactor events")
    return df


def _sentences(text):
    text = BOILERPLATE.sub(" ", text)
    text = re.sub(r"\s+", " ", text.replace('"', " ")).strip()
    return re.split(r"(?<=[.;!?])\s+(?=[A-Z0-9\[])", text)


def label_event(text):
    """Returns (device_class or None, snippet, n_device_classes_mentioned).
    Only labels when exactly one device class is mentioned AND failure
    language appears in the snippet."""
    found = {k for k, rx in _COMPILED.items() if rx.search(text or "")}
    if len(found) != 1:
        return None, None, len(found)
    device = found.pop()
    sents = _sentences(text)
    for i, s in enumerate(sents):
        if _COMPILED[device].search(s):
            snippet = s
            # short mention sentence: include the next one (often the cause)
            if len(snippet) < 120 and i + 1 < len(sents):
                snippet = snippet + " " + sents[i + 1]
            snippet = snippet[:400]
            if FAILURE_WORDS.search(snippet):
                return device, snippet, 1
            return None, None, 1
    return None, None, 1


def build_nrc_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw NRC events -> the shared event schema plus site metadata.
    All events are kept; only cleanly labeled ones get a field_device."""
    rows = []
    for _, r in raw.iterrows():
        text = r["event_text"] if isinstance(r["event_text"], str) else ""
        device, snippet, n_classes = label_event(text)
        date = pd.to_datetime(r["event_date"], format="%m/%d/%Y", errors="coerce")
        if pd.isna(date):
            date = pd.to_datetime(r["notification_date"], format="%m/%d/%Y", errors="coerce")
        title = text.split("\n")[0][:160]
        init_p, curr_p = r.get("initial_power"), r.get("current_power")
        rows.append({
            "date": date.date() if pd.notna(date) else None,
            "month": date.strftime("%b-%y") if pd.notna(date) else None,
            "mins": None,  # NRC reports don't carry a delay duration
            "reason_text": snippet if snippet else title,
            "area": "GRID_OFFSITE_POWER" if GRID_PATTERN.search(text) else "REACTOR_PLANT",
            "category": "Field Device" if device else "Non-Field Device",
            "field_device": device,
            "tag_source": "nrc_keyword_weak_label" if device else "nrc_unlabeled",
            "source_file": f"NRC EN {r['event_number']}",
            "site": str(r["facility"] or "UNKNOWN").upper(),
            "plant_type": "NUCLEAR_POWER",
            "state": r.get("state"),
            "country": "USA",
            "reactor_trip": bool(r.get("scram")) or bool(TRIP_PATTERN.search(text)),
            "grid_related": bool(GRID_PATTERN.search(text)),
            "power_lost_pct": (float(init_p) - float(curr_p))
                              if pd.notna(init_p) and pd.notna(curr_p) else None,
            "n_device_classes_mentioned": n_classes,
            "failure_modes": _tags(text, _FAILURE_MODES) if device else None,
            "grid_causes": _tags(text, _GRID_CAUSES) if GRID_PATTERN.search(text) else None,
            "event_title": title,
        })
    df = pd.DataFrame(rows)
    df = df[df["date"].notna()].sort_values("date").reset_index(drop=True)
    labeled = df["field_device"].notna().sum()
    logger.info(f"NRC labeled: {labeled} of {len(df)} events "
                f"({(df['n_device_classes_mentioned'] >= 2).sum()} ambiguous multi-device events left unlabeled)")
    logger.info("Label distribution:\n" + df["field_device"].value_counts().to_string())
    return df


# ------------------------------------------------------------------ WRI --
WRI_COLUMNS = ["country", "country_long", "name", "gppd_idnr", "capacity_mw", "latitude",
               "longitude", "primary_fuel", "commissioning_year", "owner",
               "estimated_generation_gwh_2017"]


def fetch_wri_power_plants() -> pd.DataFrame:
    logger.info("Downloading WRI Global Power Plant Database...")
    raw = _fetch(WRI_URL, timeout=120)
    if raw is None:
        raise RuntimeError("Could not download the WRI Global Power Plant Database.")
    df = pd.read_csv(io.StringIO(raw), low_memory=False)
    df = df[[c for c in WRI_COLUMNS if c in df.columns]]
    logger.info(f"WRI: {len(df)} plants, {df['capacity_mw'].sum() / 1000:.0f} GW across "
                f"{df['country'].nunique()} countries")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--relabel", action="store_true", help="skip downloads, re-label cached raw NRC data")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests per worker")
    args = ap.parse_args()

    existing = pd.read_csv(NRC_RAW_PATH) if NRC_RAW_PATH.exists() else None
    if args.relabel and existing is not None:
        raw = existing
    else:
        raw = fetch_nrc_events(dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end),
                               workers=args.workers, delay=args.delay, existing=existing)
        raw.to_csv(NRC_RAW_PATH, index=False, compression="gzip")
        logger.info(f"Saved raw NRC events to {NRC_RAW_PATH}")

    events = build_nrc_events(raw)
    events.to_csv(config.EXTERNAL_EVENTS_PATH, index=False, compression="gzip")
    logger.info(f"Saved {len(events)} external events to {config.EXTERNAL_EVENTS_PATH}")

    if not WRI_PATH.exists():
        wri = fetch_wri_power_plants()
        wri.to_csv(WRI_PATH, index=False, compression="gzip")
        logger.info(f"Saved WRI plants to {WRI_PATH}")


if __name__ == "__main__":
    main()
