# Competition reference — WiDS Global Datathon 2026

A condensed reference for anyone reading this repo without the Kaggle page open.
Authoritative details live on the
[competition page](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26).

## Task

Predict how soon a wildfire will threaten an evacuation zone, using only signals available
in the first five hours after the initial perimeter observation (`t0`).

- **Target definition** — time from `t0 + 5h` until the fire comes within 5 km of any
  evacuation zone centroid.
- **Observation window** — 72 hours.
- **Submission** — for each `event_id`, four cumulative hit probabilities: `prob_12h`,
  `prob_24h`, `prob_48h`, `prob_72h`.

The framing is right-censored survival analysis rather than binary classification, because
operational decisions are time-bound and comparative: not just *will this fire be dangerous*,
but *how soon*, *how confident*, and *which incident goes to the front of the queue*.

## Censoring

| `event` | `time_to_hit_hours` | Meaning |
|---|---|---|
| 1 | Observed hit time | Fire reached the 5 km threshold inside the window |
| 0 | Last observation (≤ 72h) | Fire had not reached the threshold when observation ended |

## Metric

```
Hybrid = 0.3 · C-index + 0.7 · (1 − WeightedBrier)
WeightedBrier = 0.3·Brier@24h + 0.4·Brier@48h + 0.3·Brier@72h
```

**C-index (30%)** — how well fires are ranked by urgency. Range 0.5 to 1.0, higher better.

**Weighted Brier (70%)** — calibration, evaluated censor-aware:

| Row at horizon `H` | Treatment |
|---|---|
| Hit by `H` | Label 1 |
| Censored after `H` | Label 0 |
| Censored before `H` | **Excluded** — outcome unknown |

48h carries the heaviest weight because 24–48 hours is the strongest operational value zone:
enough lead time to act, close enough to be urgent. 72h is included for extended planning but
is less immediate.

## Submission validation

- Exact schema: `event_id, prob_12h, prob_24h, prob_48h, prob_72h`
- IDs must match the test set exactly — no missing, extra, or duplicate IDs
- All probabilities in `[0, 1]`
- Monotonic row-wise: `prob_12h ≤ prob_24h ≤ prob_48h ≤ prob_72h`

## Dataset shape

| | Rows | Columns |
|---|---|---|
| `train.csv` | 221 | 37 (34 features + `event_id`, `event`, `time_to_hit_hours`) |
| `test.csv` | 95 | 35 (34 features + `event_id`) |

Event rate: 69 hits / 152 censored — **31.2%**.

Both the small sample size and the heavy censoring drive the modelling choices in this repo:
survival models and IPCW to use the censored rows rather than discard them, and heavy seed
averaging because single-seed results are unstable at this scale.

## Timeline

| Date | Milestone |
|---|---|
| 28 Jan 2026 | Competition opens |
| 24 Apr 2026 | Entry and team merger deadline |
| 1 May 2026 | Final submission deadline |

## Organisers

Run by [Women in Data Science (WiDS)](https://www.widsconference.org/) in partnership with
[Watch Duty](https://www.watchduty.org/), a nonprofit delivering hyperlocal real-time wildfire
alerts across 22 US states, and hosted on Kaggle.
