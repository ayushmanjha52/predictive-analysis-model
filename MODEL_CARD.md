# Model Card — Combi Mill Field-Device Delay Classifier

## What this model does
Classifies which field device (Photocell, HMD, Proximity, LVDT,
Encoder, Pressure Switch, Flow Switch, or Laser) most likely caused a
given delay, from the free-text description in the delay log. Also
provides a 6-month volume forecast per device, area-level risk
ranking, and fusion with FMEA risk ratings where available.

This model directly supports the FY27 target: **Reliability
Enhancement of field devices by reduction of delays by 50% in FY27
over FY26.**

## Current production model
- **Algorithm: CatBoost** (selected automatically by comparing Random
  Forest, HistGradientBoosting, Logistic Regression, and CatBoost --
  see "How the winner is chosen" below)
- **Honest holdout accuracy: 69.4%** -- trained on Nov'25-Apr'26,
  tested only on May'26 (a month the model never saw during training
  or feature-fitting)
- **Training samples: 344** labeled events across 8 device classes
- **CV accuracy: 75.3%** -- reported for reference only; NOT the
  headline number (see "Why holdout, not CV" below)

## How the winner is chosen
Every training run compares 3-4 candidate model families and selects
the winner by **holdout accuracy**, not cross-validation score. This
project has repeatedly confirmed CV can overstate real-world
performance by 10-25 points on this kind of monthly-batched
maintenance data -- a model can score well on shuffled cross-validation
folds while genuinely failing to generalize to a truly unseen future
month.

Because the comparison runs fresh each time `train.py` executes,
**the actual winning algorithm can differ between environments**
depending on what's installed (CatBoost is optional and only competes
when the `catboost` package is present). This is intentional: the
system always deploys whichever model performs best on the honest
holdout for that specific run, not a hardcoded choice.

| Candidate | CV Accuracy | Holdout Accuracy |
|---|---|---|
| CatBoost | 75.3% | 69.4% (winner) |
| Random Forest | 76.5% | 66.7% |
| Logistic Regression | 70.1% | 66.7% |
| HistGradientBoosting | 66.6% | 52.8% |

Note the Random Forest vs. CatBoost gap: Random Forest scores higher
on CV (76.5%) but lower on the honest holdout (66.7%) -- exactly the
CV-overstates-performance pattern this project selects against.

## Why holdout, not CV
Cross-validation shuffles all months together, so a fold's "test" data
can share close vocabulary with adjacent months in the "training" data
-- this leaks information a real future month won't have. The holdout
test trains on Nov'25-Apr'26 only and evaluates strictly on May'26,
which the model has never seen in any form. This is the number
reported to leadership, and the number this model card treats as the
model's real-world accuracy: 69.4%.

## Class distribution (known imbalance)
| Device | Training samples | Share |
|---|---|---|
| Photocell | 103 | 29.9% |
| HMD | 61 | 17.7% |
| Proximity | 58 | 16.9% |
| LVDT | 44 | 12.8% |
| Encoder | 36 | 10.5% |
| Pressure Switch | 25 | 7.3% |
| Flow Switch | 11 | 3.2% |
| Laser | 6 | 1.7% |

Flow Switch and Laser have few examples; their per-class accuracy is
less statistically reliable than Photocell's or HMD's. RFID, TT, and
HIP (1 event each) are excluded from training entirely (below the
5-sample minimum) and are tracked separately in the dashboard's "Other
Devices -- Low Sample Size" table, never mixed into ranked comparisons.

## Known weak point
Encoder has been the consistently weakest class across every model
family tried in this project. This is a real, evidence-based
limitation (confirmed via confusion matrix and holdout
misclassifications, including a real production case where "encoder
feedback fault" was predicted as LVDT) -- not something more code
alone can fix. Practically: treat Encoder predictions with more
scrutiny than other devices until shadow-mode data accumulates
specifically for this class.

## Data coverage
Of all 682 logged delay events, 51% (347) resolve to one of the 8
tracked field devices; the remaining 49% are either genuine
non-field-device causes (crane, motor, furnace, sequence issues -- all
legitimate) or delays whose description didn't name a device clearly
enough to auto-tag. See LOGGING_GUIDE.md for how this gap is being
addressed going forward, and the dashboard's "Untagged Delay Review
Queue" for prioritizing manual review of the highest-impact unresolved
delays.

## Validation status
This model has NOT yet been validated in shadow mode against real,
independent human diagnosis at scale. The shadow-mode mechanism
(/shadow_predict, /shadow_resolve, /shadow_stats, and the dashboard's
"Shadow-Mode Review" form) is built and operational, but as of this
writing has zero resolved entries. Do not treat this model's
predictions as authoritative for unsupervised maintenance decisions
until a meaningful sample of shadow-mode agreement data has been
collected and reviewed.

## How to regenerate this card
Numbers above reflect the training run described in
models/training_manifest.json as of this document's last edit. After
any retrain (python src/train.py or retrain_model.py), regenerate this
section from the new manifest -- do not leave stale numbers here; a
model card that describes a different model than what's actually
deployed is actively misleading.

## Intended use
Prioritization and diagnostic support for field-device maintenance
planning at the Combi Mill. Not intended, in its current validation
state, to trigger automated maintenance actions without human review.