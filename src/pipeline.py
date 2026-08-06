"""
End-to-end pipeline: load -> engineer -> train -> blend -> submit.

Run with::

    python -m src.pipeline --mode fast     # ~5 min smoke test
    python -m src.pipeline --mode full     # ~40 min, submission run
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as cfg
from src.features import ID_COL, TARGET_COLS, create_features, split_xy
from src.metrics import (
    compute_brier,
    compute_hybrid_score,
    enforce_monotonicity,
    validate_submission,
)
from src.models import train_gbsa_ensemble, train_lgb_ipcw

warnings.filterwarnings("ignore")

SUBMISSION_COLS = ["event_id", "prob_12h", "prob_24h", "prob_48h", "prob_72h"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(data_dir: Path | None = None):
    """Load the three competition CSVs."""
    data_dir = Path(data_dir) if data_dir else cfg.DATA_DIR
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")

    n_hit = int(train["event"].sum())
    print(f"train {train.shape} | test {test.shape}")
    print(f"events: {n_hit} hit / {len(train) - n_hit} censored "
          f"({n_hit / len(train):.1%} event rate)")
    return train, test, sample


# --------------------------------------------------------------------------- #
# Blending
# --------------------------------------------------------------------------- #
def blend(
    gbsa: np.ndarray,
    lgb_preds: dict[int, np.ndarray],
    w24: float = cfg.W_GBSA_24,
    w48: float = cfg.W_GBSA_48,
    w12: float = cfg.W_GBSA_12,
    power_cal_24: float = cfg.POWER_CAL_24,
    p72_mode: str = cfg.P72_MODE,
) -> np.ndarray:
    """
    Combine the survival backbone with the per-horizon classifiers.

    Steps
    -----
    1. Power-calibrate the 24h GBSA column (``p -> p ** alpha``) to correct mild
       over-confidence.
    2. Convex-blend GBSA and LightGBM per horizon.
    3. Set the 72h column (see ``config.P72_MODE`` for the rationale).
    4. Repair monotonicity across horizons.
    """
    out = gbsa.copy()
    if power_cal_24 != 1.0:
        out[:, 1] = np.clip(out[:, 1] ** power_cal_24, 0, 1)

    blended = out.copy()
    blended[:, 0] = w12 * out[:, 0] + (1 - w12) * lgb_preds[12]
    blended[:, 1] = w24 * out[:, 1] + (1 - w24) * lgb_preds[24]
    blended[:, 2] = w48 * out[:, 2] + (1 - w48) * lgb_preds[48]

    if p72_mode == "constant1":
        blended[:, 3] = 1.0
    elif p72_mode != "model":
        raise ValueError(f"Unknown P72_MODE: {p72_mode!r}")

    return enforce_monotonicity(blended)


def report_oof(time_values, event_values, oof_final: np.ndarray) -> float:
    """Print the full OOF diagnostic block and return the hybrid score."""
    hybrid, c_idx, wbrier = compute_hybrid_score(
        time_values, event_values, oof_final[:, 1], oof_final[:, 2], oof_final[:, 3]
    )
    briers = {
        h: compute_brier(time_values, event_values, oof_final[:, i], h)
        for i, h in enumerate(cfg.HORIZONS_PRED)
    }

    print("=" * 62)
    print(f"OOF hybrid  {hybrid:.5f}   C-index {c_idx:.4f}   WBrier {wbrier:.5f}")
    print("  " + "  ".join(f"B{h}={briers[h]:.5f}" for h in cfg.HORIZONS_PRED))
    print("=" * 62)
    return hybrid


# --------------------------------------------------------------------------- #
# Blend-weight search
# --------------------------------------------------------------------------- #
def grid_search_weights(
    gbsa_oof_raw: np.ndarray,
    lgb_oof: dict[int, np.ndarray],
    time_values: np.ndarray,
    event_values: np.ndarray,
    baseline: float,
) -> dict:
    """
    Small grid over blend weights and the 24h calibration exponent.

    On 221 rows a large apparent gain is a warning sign, not a win: the search
    is being fit to fold noise. We only trust improvements below
    ``config.GRID_TRUST_THRESHOLD`` and otherwise keep the incumbent settings.
    """
    best_score, best_params = -np.inf, None

    for w24 in cfg.GRID_W24:
        for w48 in cfg.GRID_W48:
            for pcal in cfg.GRID_PCAL:
                arr = blend(gbsa_oof_raw, lgb_oof, w24=w24, w48=w48, power_cal_24=pcal)
                score, _, _ = compute_hybrid_score(
                    time_values, event_values, arr[:, 1], arr[:, 2], arr[:, 3]
                )
                if score > best_score:
                    best_score = score
                    best_params = {"W_GBSA_24": w24, "W_GBSA_48": w48,
                                   "POWER_CAL_24": pcal}

    gain = best_score - baseline
    print(f"best OOF {best_score:.5f} vs baseline {baseline:.5f} (gain {gain:+.5f})")
    print(f"best params: {best_params}")
    if gain > cfg.GRID_TRUST_THRESHOLD:
        print(f"gain > {cfg.GRID_TRUST_THRESHOLD} -> likely overfitting the OOF "
              f"folds; NOT adopted")
        adopt = False
    elif gain > 0:
        print(f"gain < {cfg.GRID_TRUST_THRESHOLD} -> small and stable; safe to adopt")
        adopt = True
    else:
        print("no improvement; keeping current weights")
        adopt = False

    return {"score": best_score, "params": best_params, "adopt": adopt}


# --------------------------------------------------------------------------- #
# Submission
# --------------------------------------------------------------------------- #
def build_submission(
    test_df: pd.DataFrame, preds: np.ndarray, sample_sub: pd.DataFrame
) -> pd.DataFrame:
    """Assemble, re-order to the sample submission, and validate."""
    sub = pd.DataFrame(
        {
            "event_id": test_df[ID_COL].values,
            "prob_12h": preds[:, 0],
            "prob_24h": preds[:, 1],
            "prob_48h": preds[:, 2],
            "prob_72h": preds[:, 3],
        }
    )
    sub = sample_sub[[ID_COL]].merge(sub, on=ID_COL, how="left")
    validate_submission(sub, sample_sub)
    return sub


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(run_mode: str = cfg.RUN_MODE, do_oof: bool = cfg.DO_OOF) -> pd.DataFrame:
    from sksurv.util import Surv

    t_start = time.time()
    gbsa_seeds, lgb_seeds = cfg.get_seeds(run_mode)
    print(f"run mode: {run_mode} | GBSA seeds {len(gbsa_seeds)} | "
          f"LGBM seeds {len(lgb_seeds)} | OOF {do_oof}\n")

    train_df, test_df, sample_sub = load_data()

    # GBSA consumes the raw columns: the survival trees find the interactions
    # themselves, and the engineered frame measurably hurt it in ablation.
    X_surv_train = split_xy(train_df, is_train=True)
    X_surv_test = split_xy(test_df, is_train=False)

    # LightGBM consumes the engineered frame, where the explicit ETA and
    # threat-score terms do help the shallow per-horizon classifiers.
    X_lgb_train = split_xy(create_features(train_df), is_train=True)
    X_lgb_test = split_xy(create_features(test_df), is_train=False)

    time_values = train_df["time_to_hit_hours"].values
    event_values = train_df["event"].values
    y_surv = Surv.from_arrays(
        event=train_df["event"].astype(bool), time=train_df["time_to_hit_hours"]
    )
    print(f"GBSA features {X_surv_train.shape[1]} | "
          f"LGBM features {X_lgb_train.shape[1]}\n")

    gbsa_oof, gbsa_test = train_gbsa_ensemble(
        X_surv_train, y_surv, X_surv_test, event_values,
        seeds=gbsa_seeds, do_oof=do_oof,
    )
    print()
    lgb_oof, lgb_test = train_lgb_ipcw(
        X_lgb_train, X_lgb_test, time_values, event_values, seeds=lgb_seeds
    )
    print()

    if do_oof:
        oof_final = blend(gbsa_oof, lgb_oof)
        baseline = report_oof(time_values, event_values, oof_final)
        print()
        grid_search_weights(gbsa_oof, lgb_oof, time_values, event_values, baseline)
        print()

    test_final = blend(gbsa_test, lgb_test)
    sub = build_submission(test_df, test_final, sample_sub)

    cfg.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(cfg.OUTPUT_PATH, index=False)
    print(f"saved -> {cfg.OUTPUT_PATH}  ({time.time() - t_start:.0f}s total)")
    print(sub.head())
    return sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WiDS 2026 wildfire survival pipeline")
    parser.add_argument("--mode", choices=["fast", "full"], default=cfg.RUN_MODE)
    parser.add_argument("--no-oof", action="store_true",
                        help="skip OOF computation (roughly halves runtime)")
    args = parser.parse_args()
    main(run_mode=args.mode, do_oof=not args.no_oof)
