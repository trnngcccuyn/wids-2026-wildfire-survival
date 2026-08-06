# Wildfire Evacuation-Threat Forecasting — WiDS Global Datathon 2026

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-WiDS%202026-20BEFF.svg)](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)

> **Final placement: 210 / 1,754 teams — top 12%**
> Right-censored survival modelling of how quickly a wildfire threatens an evacuation zone,
> using only the first five hours after ignition.

*[Đọc bản tiếng Việt →](README.vi.md)*

---

## The problem

When a wildfire ignites, an incident commander must decide which communities to warn, when
to warn them, and where to send scarce crews — before anything is certain. Most wildfire
models answer a single yes/no question: *will this fire become dangerous?* Emergency response
needs more than that. It needs **how soon**, **how confident**, and **which fire first**.

This competition, run with [Watch Duty](https://www.watchduty.org/), frames that operational
need as survival analysis. Given features computed strictly from the first five hours after
the initial perimeter observation (`t0`), predict the probability that a fire comes within
**5 km of an evacuation zone centroid** by 12h, 24h, 48h, and 72h after `t0 + 5h`.

The target is right-censored:

| | |
|---|---|
| `event = 1` | Fire hit within the 72h window; `time_to_hit_hours` is the observed hit time |
| `event = 0` | Censored; `time_to_hit_hours` is the last observation in the window (≤ 72h) |

### Scoring

```
Hybrid = 0.3 · C-index + 0.7 · (1 − WeightedBrier)
WeightedBrier = 0.3·Brier@24h + 0.4·Brier@48h + 0.3·Brier@72h
```

Brier is **censor-aware**: fires censored *before* a horizon are excluded from that horizon,
because their outcome is genuinely unknown. The 70/30 split means **calibration dominates
ranking** — probabilities an emergency manager could threshold on matter more than a
correct ordering. That single fact shaped most of the design below.

---

## Results

| Metric | Out-of-fold |
|---|---|
| **Hybrid score** | **0.97476** |
| C-index | 0.9456 |
| Weighted Brier | 0.01274 |
| Brier @ 12h | 0.05060 |
| Brier @ 24h | 0.02674 |
| Brier @ 48h | 0.01180 |
| Brier @ 72h | 0.00000 |

OOF ≈ 0.9748 corresponded to roughly 0.970 on the public leaderboard — a gap that stayed
stable across submissions, which is what let us trust local validation for decisions instead
of burning submission slots.

---

## Approach

Two model families with complementary failure modes, blended per horizon.

```
raw features (34) ──► GBSA ensemble          ┐
                      10 configs × 40 seeds   │
                      × 5 folds = 2,000 fits  │
                                              ├─► per-horizon blend
engineered (54) ────► LightGBM + IPCW        │   + 24h power calibration
                      one model per horizon   │   + 72h constant
                      × 25 seeds × 5 folds   ┘   + monotonicity repair
                                                        │
                                                        ▼
                                            p12 ≤ p24 ≤ p48 ≤ p72
```

### A. Gradient Boosting Survival Analysis — the backbone

Handles right-censoring natively, which matters when **152 of 221 training fires never hit**.
One fit yields the whole survival curve, so all four horizons come out internally consistent
instead of needing four models stitched together. Ten deliberately diverse configurations
(depth 2–4, low learning rates) — diversity *across* configs stabilised the ensemble more
than tuning any single config well.

### B. LightGBM + inverse-probability-of-censoring weighting — the calibration fix

One binary classifier per horizon (12h / 24h / 48h), trained only on rows whose outcome at
that horizon is known, with IPCW weights to undo the resulting selection bias. Dropping
censored rows outright would bias the model — fires that leave observation early are not a
random sample — so each retained row is up-weighted by `1/G(t)`, with `G` the Kaplan–Meier
estimate of the *censoring* survival function.

This family optimises exactly what the Brier score measures, at exactly the horizon it is
measured. It earns the majority of the blend at 48h (0.55), the heaviest-weighted term in
the metric.

### Blend weights

| Horizon | GBSA | LightGBM | Note |
|---|---|---|---|
| 12h | 0.97 | 0.03 | Survival model near-unbeatable; sparsest target |
| 24h | 0.95 | 0.05 | Plus power calibration `p → p^1.1` |
| 48h | 0.45 | 0.55 | Classifier wins where the metric weighs most |
| 72h | — | — | Constant 1.0 — see below |

### Two decisions worth explaining

**Feature asymmetry.** GBSA is fed the *raw* columns; LightGBM the *engineered* frame. In
ablation the engineered features measurably hurt GBSA — the survival trees find those
interactions themselves and the extra collinear columns only add variance — while they
clearly helped the shallow per-horizon classifiers.

**The 72h constant.** Under the censor-aware rule, Brier@72h scores only fires that hit by
72h (label 1) and fires censored *after* 72h (label 0). In this dataset the second group is
empty — censoring occurs at 72h, not past it — so every row surviving the mask is a hit.
Predicting 1.0 gives Brier@72h = 0.0 exactly, worth ≈ 0.021 of hybrid score, and cannot
disturb the C-index since a constant shifts every fire's risk equally. **This is fitted to a
quirk of this dataset's censoring mechanism, not to wildfire physics.** An operational system
would emit a real 72h probability. We flag it rather than dress it up as insight.

### The most valuable line of code

```python
if gain > 0.001:
    print("likely fitting fold noise → NOT adopted")
```

On 221 rows, a large apparent gain from re-tuning three blend parameters is a warning sign,
not a win. We only adopted small, stable OOF improvements. This rule cost a few tenths of a
point on paper several times and saved us from at least two shakeup candidates.

---

## Repository structure

```
.
├── notebooks/
│   └── wids2026_wildfire_survival.ipynb   Full annotated solution + EDA (Kaggle-runnable, self-contained)
├── src/
│   ├── config.py                          Every tunable constant in one place
│   ├── features.py                        Feature engineering
│   ├── metrics.py                         C-index, censor-aware Brier, IPCW, monotonicity
│   ├── models.py                          GBSA ensemble + LightGBM IPCW trainers
│   └── pipeline.py                        Load → train → blend → submit
├── tests/
│   └── test_smoke.py                      Synthetic-data tests; no competition CSVs needed
├── docs/
│   └── COMPETITION.md                     Task, metric, and timeline summary
├── data/                                  Place competition CSVs here (git-ignored)
├── submissions/                           Generated submissions (git-ignored)
├── requirements.txt
├── LICENSE
└── README.md
```

The notebook is self-contained so it runs on Kaggle as a single upload. `src/` is the same
logic refactored into importable modules, for anyone who wants to extend it.

---

## Getting started

```bash
git clone https://github.com/<your-username>/wids-2026-wildfire-survival.git
cd wids-2026-wildfire-survival

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the competition data from the
[Kaggle competition page](https://www.kaggle.com/competitions/WiDSWorldWide_GlobalDathon26)
and place `train.csv`, `test.csv`, and `sample_submission.csv` in `data/`.

```bash
# Verify the install — synthetic data, no competition CSVs required (~1 min)
python -m pytest tests/ -v

# Smoke run: 10 seeds, ~5 minutes
python -m src.pipeline --mode fast

# Submission run: 40 GBSA seeds / 25 LightGBM seeds, ~40 minutes on Kaggle CPU
python -m src.pipeline --mode full
```

The submission is written to `submissions/submission.csv` (or `/kaggle/working/` when run on
Kaggle) and is validated against every rule the competition checker enforces — schema, exact
ID match, `[0, 1]` range, and row-wise monotonicity — before it hits disk.

### Runtime

| Mode | Seeds (GBSA / LGBM) | Model fits | Approx. runtime |
|---|---|---|---|
| `fast` | 10 / 10 | 500 + 300 | ~5 min |
| `full` | 40 / 25 | 2,000 + 750 | ~40 min |

CPU only — no GPU needed. Peak memory stays under 2 GB.

---

## What we would do next

- **Conformal prediction intervals** per horizon. Anyone acting on a threshold needs to know
  when the model is uncertain, not just what it predicts.
- **A real 72h model**, so the pipeline generalises past this dataset's censoring quirk.
- **Spatial cross-validation by region**, to check the model is not leaning on geography-
  specific patterns that would not transfer to a new fire season.
- **Cost-sensitive thresholds** — a missed evacuation and a false alarm are nowhere near
  symmetric in cost, and the metric treats them as if they were.

---

## Team

Four of us, working across the whole pipeline rather than splitting it into owned pieces.

| Name | Links |
|---|---|
| Tran Ngoc Cac Uyen | [GitHub](https://github.com/trnngcccuyn) |
| Nguyen Hong Linh | [GitHub](https://github.com/HLiuga05) |
| Nguyen Dieu Le | [GitHub](https://github.com/dieule-0810) |
| Le Minh Phuc Tien | [GitHub](https://github.com/TienLe-0207) |

---

## Acknowledgements

- **[Watch Duty](https://www.watchduty.org/)** — the nonprofit behind the data and the problem
  framing. They deliver real-time wildfire alerts to millions of people across the US, run by a
  small staff alongside hundreds of volunteer firefighters, dispatchers, and radio operators.
- **[Women in Data Science (WiDS)](https://www.widsconference.org/)** for organising the Global
  Datathon, and **Kaggle** for hosting it.
- Built on [scikit-survival](https://scikit-survival.readthedocs.io/) and
  [LightGBM](https://lightgbm.readthedocs.io/).

## License

Code released under the [MIT License](LICENSE). The competition dataset is **not** included
and remains subject to the WiDS Datathon / Kaggle competition rules.
