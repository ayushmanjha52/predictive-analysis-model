# Model Card — Field-Device Delay Classifier (Combi Mill + Power Fleet)

## What this model does
Classifies which field device (Photocell, HMD, Proximity, LVDT, Encoder,
Pressure Switch, Flow Switch, or Laser) most likely caused a delay, using
the free-text description in the delay log. The system around it also
provides a 6-month volume forecast per device, area-level risk ranking,
fusion with FMEA risk ratings, and **fleet-wide benchmarking against
public power-plant data**.

This supports the FY27 target: **Reliability Enhancement of field devices
by reduction of delays by 50% in FY27 over FY26.**

## Current production model
Source of truth: `models/training_manifest.json` (trained 2026-10-03).

- **Algorithm: Logistic Regression** + **explicit-device-mention rule**
  (both selected automatically, see "How the winner is chosen")
- **Training data: Combi Mill only** (344 hand-tagged events, 8 classes).
  Public power-plant data was evaluated and *not* adopted; see below.
- **Honest backtest accuracy: 62.7%** (161 test events). The model is
  trained only on months before each test month, then tested on
  Mar-26, Apr-26 and May-26 in turn:

  | Test month | Accuracy |
  |---|---|
  | Mar-26 | 77.8% |
  | Apr-26 | 56.1% |
  | May-26 | **75.0%** (latest month; the old card's comparable number was 69.4%) |

- **CV accuracy (home data): 78.5%.** Reported for reference only. Shuffled
  CV overstates real performance on this data by roughly 15 points.

### Why the headline changed from "69.4%" to "62.7%"
The old headline was one month (May-26, 36 events). One misclassification
there moves the number about 3 points, so choosing between models a few
points apart was mostly noise. The new headline pools **three** unseen
months with the same honest procedure. That is a harder and more stable
number, not a worse model: on the same May-26 month the new model scores
**75.0% vs. 69.4%**.

## What changed in this version, and what each change was worth
All figures are pooled backtest accuracy on the same 161 real test
events. Every change had to earn its place on this number.

| Change | Effect |
|---|---|
| Word + **character** n-gram TF-IDF (handles misspellings like "continous", "missisng", "proxy") | baseline for this version: 59.0% (LR), 60.2% (CatBoost) |
| **Explicit-device-mention rule**: if the text names exactly one device class, use it | **+1.9 to +6.8 pts on every one of 16 model/data configs** (+3.7 on the winner); adopted (62.7%) |
| Restricting home predictions to classes the home site has seen | built in; prevents power-plant-only labels (e.g. LEVEL_TRANSMITTER) at the mill |
| Excluding `/log_event` rows from training (their label is the model's own guess) | correctness fix; prevents the model training on its own predictions |
| **Public power-plant data** (NRC events, down-weighted) | best result 64.0% vs. 62.7% (+1.2 pts, about 2 test events). **Below the 2-pt bar, so not adopted** |

The explicit-mention rule covers 29% of home events. On those events the
named device matches the human label 94% of the time.

### On "more data": what the power-plant data did and did not do
More *relevant labeled* data is what improves a classifier. The public
NRC power-reactor reports added about 3,900 real events, but plant staff
rarely name a specific sensor in them. Only about 1–2% of events name
exactly one field-device class, and most of those are classes the mill
doesn't have (transmitters, level switches). Training with them gave a
gain within noise, so the comparison correctly kept the home-only model.
`train.py` re-runs this comparison on every retrain: if future external
data starts helping by at least 2 points, it is adopted automatically.

The power-plant data is used for **insights** (Power Fleet tab): failure
modes, trip rates, grid events, and plant benchmarking. It is not used
to inflate the accuracy number. External rows are never in any test set.

**The fastest path to higher accuracy is more hand-tagged mill data.**
Resolve shadow-mode entries and tag the "Untagged Delay Review Queue".
335 untagged mill events (12,105 min) are waiting there.

## How the winner is chosen
Each run of `python src/train.py` compares:
- 4 model families (Random Forest, HistGradientBoosting, Logistic
  Regression, CatBoost if installed)
- × 4 data mixes (home only; home + external at weight 0.15 / 0.4 / 1.0)
- × explicit-mention rule on/off

The run selects by pooled honest backtest. Ties go to the simpler option.
External data must win by at least 2 points.

## Per-class performance (honest backtest)
| Device | Precision | Recall | Test events |
|---|---|---|---|
| Pressure Switch | 0.90 | 1.00 | 9 |
| HMD | 0.84 | 0.79 | 33 |
| Photocell | 0.72 | 0.62 | 37 |
| Proximity | 0.61 | 0.74 | 34 |
| LVDT | 0.39 | 0.48 | 23 |
| Encoder | 0.40 | 0.25 | 24 |

## Known weak point
**Encoder and LVDT** remain the weakest classes, and they are confused
with each other. The real data shows why. Kick-off arm and position-
feedback faults are logged in near-identical words whichever sensor is
at fault. Six rows read "Kick off arm **encoder** feedback missing" but
are labeled LVDT, which is worth a check with the plant team. Treat
Encoder/LVDT predictions with extra scrutiny.

## Class distribution (training)
Photocell 104 · HMD 61 · Proximity 58 · LVDT 44 · Encoder 37 ·
Pressure Switch 25 · Flow Switch 11 · Laser 6. RFID, TT, and HIP
(1 event each) are below the 5-sample minimum and are excluded.

## Data coverage
Of 684 logged mill events, 349 resolve to a field device. The rest are
genuine non-field-device causes (crane, motor, furnace, sequence) or
descriptions too vague to tag.

## Validation status
Shadow mode (`/shadow_predict`, `/shadow_resolve`, dashboard form) is
operational, with 1 resolved entry so far. Do not use this model for
unsupervised maintenance decisions until meaningful shadow-mode data
exists. **On Render's free plan, shadow entries are lost on redeploy**
(see DEPLOYMENT.md for a persistent disk).

## Intended use
Prioritization and diagnostic support for field-device maintenance at
the Combi Mill, with power-fleet context. Not intended to trigger
automated maintenance actions without human review.

## How to regenerate this card
After any retrain, update the numbers above from
`models/training_manifest.json` and `reports/holdout_report.csv`. A card
that describes a different model than the one deployed is misleading.
