"""
Model definitions and training loops.

Two complementary estimators
----------------------------
**A. Gradient Boosting Survival Analysis (GBSA)** — the backbone. It models the
full survival function, so a single fit yields all four horizons at once and is
internally consistent (monotone by construction). It handles right-censoring
natively, which matters here: 152 of 221 training fires never hit within 72h.

**B. LightGBM + IPCW** — one binary classifier per horizon (12h / 24h / 48h),
trained only on rows whose outcome at that horizon is known, with inverse-
probability-of-censoring weights to undo the resulting selection bias. It has no
proportional-hazards assumption to violate and calibrates the 48h column
noticeably better than GBSA does.

Both are averaged over many seeds and 5 stratified folds. On 221 rows that
averaging is not a nicety — single-seed OOF swings by several thousandths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import HORIZONS_PRED, N_FOLDS
from src.metrics import compute_brier, compute_ipcw_weights, make_binary_target

# --------------------------------------------------------------------------- #
# A. GBSA configurations
# --------------------------------------------------------------------------- #
# Ten deliberately diverse configurations. Depth is kept at 2-4 and learning
# rates low because the training set is tiny; the diversity across configs does
# more for stability than any single well-tuned setting.
GBSA_CONFIGS: list[dict] = [
    {"learning_rate": 0.010, "subsample": 0.70, "max_depth": 3, "min_samples_leaf": 12, "min_samples_split": 3, "n_estimators": 1200},
    {"learning_rate": 0.010, "subsample": 0.85, "max_depth": 3, "min_samples_leaf": 15, "min_samples_split": 3, "n_estimators": 1200},
    {"learning_rate": 0.010, "subsample": 0.60, "max_depth": 3, "min_samples_leaf": 12, "min_samples_split": 3, "n_estimators": 1200},
    {"learning_rate": 0.005, "subsample": 0.85, "max_depth": 3, "min_samples_leaf": 12, "min_samples_split": 3, "n_estimators": 2000},
    {"learning_rate": 0.010, "subsample": 0.85, "max_depth": 3, "min_samples_leaf": 20, "min_samples_split": 3, "n_estimators": 1400},
    {"learning_rate": 0.008, "subsample": 0.75, "max_depth": 2, "min_samples_leaf": 15, "min_samples_split": 4, "n_estimators": 1500},
    {"learning_rate": 0.015, "subsample": 0.70, "max_depth": 3, "min_samples_leaf": 10, "min_samples_split": 3, "n_estimators": 1000},
    {"learning_rate": 0.005, "subsample": 0.90, "max_depth": 3, "min_samples_leaf": 18, "min_samples_split": 5, "n_estimators": 2500},
    {"learning_rate": 0.010, "subsample": 0.80, "max_depth": 4, "min_samples_leaf": 12, "min_samples_split": 3, "n_estimators": 1200},
    {"learning_rate": 0.020, "subsample": 0.65, "max_depth": 3, "min_samples_leaf": 10, "min_samples_split": 3, "n_estimators": 800},
]

# --------------------------------------------------------------------------- #
# B. LightGBM configurations, one per horizon
# --------------------------------------------------------------------------- #
# 12h is the sparsest target (few fires hit that fast), so it gets the most
# aggressive regularisation and the shallowest trees.
LGB_CONFIGS: dict[int, dict] = {
    12: {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 200,
         "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 10,
         "reg_alpha": 1.0, "reg_lambda": 3.0, "num_leaves": 4},
    24: {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 300,
         "subsample": 0.7, "colsample_bytree": 0.7, "min_child_samples": 8,
         "reg_alpha": 0.5, "reg_lambda": 2.0, "num_leaves": 7},
    48: {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 200,
         "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 5,
         "reg_alpha": 0.1, "reg_lambda": 1.0, "num_leaves": 4},
}
LGB_HORIZONS = [12, 24, 48]


# --------------------------------------------------------------------------- #
# GBSA helpers
# --------------------------------------------------------------------------- #
def get_surv_predictions(model, X: pd.DataFrame) -> np.ndarray:
    """
    Evaluate a fitted survival model at the four competition horizons.

    ``scikit-survival`` returns step functions defined only on the observed time
    grid, so horizons are clipped into each function's domain before evaluation.

    Returns
    -------
    np.ndarray, shape (n, 4)
        Hit probabilities ``1 - S(t)`` at 12/24/48/72 hours.
    """
    surv_fns = model.predict_survival_function(X)
    preds = np.empty((len(surv_fns), len(HORIZONS_PRED)), dtype=float)
    for i, fn in enumerate(surv_fns):
        t_min, t_max = fn.domain
        preds[i, :] = fn(np.clip(HORIZONS_PRED, t_min, t_max))
    return 1.0 - preds


def train_gbsa_ensemble(
    X_train: pd.DataFrame,
    y_surv: np.ndarray,
    X_test: pd.DataFrame,
    event_values: np.ndarray,
    seeds: tuple[int, ...],
    configs: list[dict] | None = None,
    do_oof: bool = True,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Train the GBSA ensemble over ``configs x seeds x folds``.

    Folds are stratified on the event indicator so every fold keeps a
    representative hit/censored ratio — important at 69 events total.

    Returns
    -------
    (oof, test)
        ``oof`` is ``None`` when ``do_oof`` is False. Both arrays have shape
        ``(n, 4)`` and hold hit probabilities at 12/24/48/72h.
    """
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis

    configs = configs or GBSA_CONFIGS
    n_h = len(HORIZONS_PRED)

    oof = np.zeros((len(X_train), n_h)) if do_oof else None
    test = np.zeros((len(X_test), n_h))

    if verbose:
        print(f"[GBSA] {len(configs)} configs x {len(seeds)} seeds x {n_folds} folds")

    for cfg_idx, cfg in enumerate(configs, 1):
        cfg_oof = np.zeros((len(X_train), n_h)) if do_oof else None
        cfg_test = np.zeros((len(X_test), n_h))

        for seed in seeds:
            seed_oof = np.zeros((len(X_train), n_h)) if do_oof else None
            seed_test = np.zeros((len(X_test), n_h))
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            for tr_idx, va_idx in cv.split(X_train, event_values):
                model = GradientBoostingSurvivalAnalysis(**{**cfg, "random_state": seed})
                model.fit(X_train.iloc[tr_idx], y_surv[tr_idx])
                if do_oof:
                    seed_oof[va_idx] = get_surv_predictions(model, X_train.iloc[va_idx])
                seed_test += get_surv_predictions(model, X_test) / n_folds

            if do_oof:
                cfg_oof += seed_oof / len(seeds)
            cfg_test += seed_test / len(seeds)

        if do_oof:
            oof += cfg_oof / len(configs)
        test += cfg_test / len(configs)
        if verbose:
            print(f"  config {cfg_idx}/{len(configs)} done")

    return oof, test


# --------------------------------------------------------------------------- #
# LightGBM + IPCW
# --------------------------------------------------------------------------- #
def train_lgb_ipcw(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    time_values: np.ndarray,
    event_values: np.ndarray,
    seeds: tuple[int, ...],
    horizons: list[int] | None = None,
    configs: dict[int, dict] | None = None,
    n_folds: int = N_FOLDS,
    verbose: bool = True,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Train one IPCW-weighted binary classifier per horizon.

    For each horizon, rows censored before it are removed from training (their
    label is unknowable) and the survivors are re-weighted by IPCW. Test
    predictions come from a model refit on all valid rows; OOF predictions for
    the excluded rows are filled in from the last fold model purely so the OOF
    array stays aligned for blending — those entries are masked out again by the
    censor-aware Brier, so they never influence any reported score.

    Returns
    -------
    (oof_by_horizon, test_by_horizon)
    """
    import lightgbm as lgb

    horizons = horizons or LGB_HORIZONS
    configs = configs or LGB_CONFIGS

    oof_out: dict[int, np.ndarray] = {}
    test_out: dict[int, np.ndarray] = {}

    if verbose:
        print(f"[LGBM] {len(seeds)} seeds x {len(horizons)} horizons {horizons}")

    for horizon in horizons:
        y_bin, mask = make_binary_target(time_values, event_values, horizon)
        valid_idx = np.where(mask)[0]
        censored_idx = np.where(~mask)[0]
        cfg = configs[horizon]

        all_oof = np.zeros(len(X_train))
        all_test = np.zeros(len(X_test))

        for seed in seeds:
            seed_oof = np.zeros(len(X_train))
            last_model = None
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

            for tr_v, va_v in cv.split(valid_idx, y_bin[mask]):
                tr_idx, va_idx = valid_idx[tr_v], valid_idx[va_v]
                weights = compute_ipcw_weights(
                    time_values[tr_idx], event_values[tr_idx], horizon
                )
                model = lgb.LGBMClassifier(
                    **cfg, objective="binary", random_state=seed, verbose=-1
                )
                model.fit(X_train.iloc[tr_idx], y_bin[tr_idx], sample_weight=weights)
                seed_oof[va_idx] = model.predict_proba(X_train.iloc[va_idx])[:, 1]
                last_model = model

            if len(censored_idx) > 0 and last_model is not None:
                seed_oof[censored_idx] = last_model.predict_proba(
                    X_train.iloc[censored_idx]
                )[:, 1]
            all_oof += seed_oof

            weights_full = compute_ipcw_weights(
                time_values[valid_idx], event_values[valid_idx], horizon
            )
            model_full = lgb.LGBMClassifier(
                **cfg, objective="binary", random_state=seed, verbose=-1
            )
            model_full.fit(
                X_train.iloc[valid_idx], y_bin[valid_idx], sample_weight=weights_full
            )
            all_test += model_full.predict_proba(X_test)[:, 1]

        oof_out[horizon] = all_oof / len(seeds)
        test_out[horizon] = all_test / len(seeds)

        if verbose:
            brier = compute_brier(
                time_values, event_values, np.clip(oof_out[horizon], 0, 1), horizon
            )
            print(f"  LGBM {horizon:>2}h  Brier={brier:.5f}")

    return oof_out, test_out
