"""
Central configuration for the WiDS 2026 wildfire survival pipeline.

Every tunable constant lives here so that experiments can be reproduced by
diffing a single file. Values marked "proven" were selected on out-of-fold
(OOF) score and confirmed against the public leaderboard.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# On Kaggle the competition data is mounted read-only under /kaggle/input.
# Locally, drop the CSVs into ./data/ and the fallback below picks them up.
KAGGLE_DATA_DIR = Path("/kaggle/input/competitions/WiDSWorldWide_GlobalDathon26")
LOCAL_DATA_DIR = Path(__file__).resolve().parents[1] / "data"

DATA_DIR: Path = KAGGLE_DATA_DIR if KAGGLE_DATA_DIR.exists() else LOCAL_DATA_DIR

KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR: Path = (
    KAGGLE_WORKING
    if KAGGLE_WORKING.exists()
    else Path(__file__).resolve().parents[1] / "submissions"
)
OUTPUT_PATH: Path = OUTPUT_DIR / "submission.csv"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# --------------------------------------------------------------------------- #
# Run mode
# --------------------------------------------------------------------------- #
# "fast"  -> ~5 min, 10 seeds per model. Use while iterating.
# "full"  -> ~40 min on Kaggle CPU, 40 GBSA seeds / 25 LGBM seeds. Use to submit.
RUN_MODE: str = "full"

# Computing OOF predictions doubles the runtime but is the only honest way to
# tune blend weights on a 221-row dataset. Keep it on unless you are only
# regenerating a submission from already-validated settings.
DO_OOF: bool = True

N_FOLDS: int = 5

# Horizons the competition asks for, in hours.
HORIZONS_PRED: list[int] = [12, 24, 48, 72]

# Horizons that enter the weighted Brier component of the metric.
BRIER_HORIZONS: dict[int, float] = {24: 0.3, 48: 0.4, 72: 0.3}

# --------------------------------------------------------------------------- #
# Blend weights (GBSA vs LightGBM), selected on OOF
# --------------------------------------------------------------------------- #
# Short horizons are dominated by the survival model; by 48h the binary
# IPCW-weighted classifier carries slightly more of the signal.
W_GBSA_12, W_LGB_12 = 0.97, 0.03
W_GBSA_24, W_LGB_24 = 0.95, 0.05
W_GBSA_48, W_LGB_48 = 0.45, 0.55

# Power calibration applied to the 24h GBSA column (p -> p**alpha).
# alpha > 1 shrinks probabilities towards 0, which helps because the survival
# model is mildly over-confident at 24h.
POWER_CAL_24: float = 1.1

# --------------------------------------------------------------------------- #
# 72h handling
# --------------------------------------------------------------------------- #
# In this dataset every uncensored fire reaches an evacuation zone before 72h,
# and censored rows that survive past 72h are excluded from Brier@72h by the
# competition's censor-aware rule. A constant 1.0 therefore scores a perfect
# Brier@72h == 0.0 and also maximises the risk score used for the C-index.
# "constant1" is the proven setting (public LB 0.97085); "model" falls back to
# the blended model output if you want to sanity-check the assumption.
P72_MODE: str = "constant1"

# --------------------------------------------------------------------------- #
# Seeds
# --------------------------------------------------------------------------- #
# With only 221 training rows, single-seed results swing by several thousandths
# of a point. Averaging many seeds is the single highest-leverage variance
# reduction available here.
GBSA_SEEDS_FULL: tuple[int, ...] = (
    123, 456, 789, 777, 666,
    1511, 1523, 2025, 2026, 2033,
    279, 239, 70, 77, 31,
    2024, 2077, 3077, 123456, 654321,
    4640, 841, 7755, 8525, 2701,
    8817, 8864, 4085, 8919, 934,
    4746, 1699, 7401, 7826, 4098,
    2921, 1204, 2752, 8384, 1284,
)
GBSA_SEEDS_FAST: tuple[int, ...] = tuple(range(42, 52))

LGB_SEEDS_FULL: tuple[int, ...] = (
    123, 456, 789, 777, 666,
    1511, 1523, 2025, 2026, 2033,
    279, 239, 70, 77, 31,
    2024, 2077, 3077, 123456, 654321,
    2034, 2035, 2036, 1984, 1991,
)
LGB_SEEDS_FAST: tuple[int, ...] = tuple(range(42, 52))


def get_seeds(run_mode: str = RUN_MODE) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(gbsa_seeds, lgb_seeds)`` for the requested run mode."""
    if run_mode == "full":
        return GBSA_SEEDS_FULL, LGB_SEEDS_FULL
    if run_mode == "fast":
        return GBSA_SEEDS_FAST, LGB_SEEDS_FAST
    raise ValueError(f"Unknown RUN_MODE: {run_mode!r} (expected 'fast' or 'full')")


# --------------------------------------------------------------------------- #
# Grid search space (Section 9 of the notebook)
# --------------------------------------------------------------------------- #
# Guard rail: on 221 rows, an OOF gain larger than ~0.001 from re-tuning blend
# weights is far more likely to be noise than a real improvement. We only adopt
# small, stable gains.
GRID_W24 = [0.90, 0.93, 0.95, 0.97]
GRID_W48 = [0.40, 0.43, 0.45, 0.48]
GRID_PCAL = [1.0, 1.05, 1.10, 1.15]
GRID_TRUST_THRESHOLD = 0.001
