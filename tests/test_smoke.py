"""
Smoke tests.

The competition data cannot be redistributed, so these tests build a small
synthetic dataset with the same schema and exercise the whole pipeline:
feature engineering -> both model families -> blending -> submission validation.

Run with::

    python -m pytest tests/ -v
    python tests/test_smoke.py       # also works without pytest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import create_features, split_xy  # noqa: E402
from src.metrics import (  # noqa: E402
    compute_brier,
    compute_c_index,
    compute_hybrid_score,
    compute_ipcw_weights,
    enforce_monotonicity,
    make_binary_target,
    validate_submission,
)
from src.models import train_gbsa_ensemble, train_lgb_ipcw  # noqa: E402
from src.pipeline import blend, build_submission  # noqa: E402

RAW_COLUMNS = [
    "dist_min_ci_0_5h", "closing_speed_m_per_h", "radial_growth_rate_m_per_h",
    "num_perimeters_0_5h", "area_first_ha", "area_growth_rate_ha_per_h",
    "alignment_abs", "event_start_month", "event_start_hour",
    "relative_growth_0_5h", "projected_advance_m", "centroid_displacement_m",
    "centroid_speed_m_per_h", "closing_speed_abs_m_per_h", "area_growth_abs_0_5h",
]


def make_synthetic(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame whose hit times depend on distance and closing speed."""
    rng = np.random.default_rng(seed)
    dist = rng.lognormal(mean=8.5, sigma=1.0, size=n)
    speed = rng.gamma(2.0, 120.0, size=n)

    df = pd.DataFrame({
        "event_id": np.arange(n),
        "dist_min_ci_0_5h": dist,
        "closing_speed_m_per_h": speed,
        "radial_growth_rate_m_per_h": rng.gamma(2.0, 40.0, size=n),
        "num_perimeters_0_5h": rng.integers(1, 6, size=n),
        "area_first_ha": rng.lognormal(2.0, 1.2, size=n),
        "area_growth_rate_ha_per_h": rng.gamma(2.0, 5.0, size=n),
        "alignment_abs": rng.uniform(0, 1, size=n),
        "event_start_month": rng.integers(1, 13, size=n),
        "event_start_hour": rng.integers(0, 24, size=n),
    })
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = rng.normal(size=n)

    eta = dist / np.maximum(speed, 1.0)
    hit_time = eta * rng.lognormal(0.0, 0.5, size=n)
    df["event"] = (hit_time <= 72).astype(int)
    df["time_to_hit_hours"] = np.minimum(hit_time, 72.0)
    return df


# --------------------------------------------------------------------------- #
def test_feature_engineering_is_finite_and_leak_free():
    df = make_synthetic()
    out = create_features(df)
    assert np.isfinite(out.select_dtypes("number").to_numpy()).all()
    assert "eta_effective" in out.columns
    assert not any(c in out.columns for c in
                   ["projected_advance_m", "centroid_speed_m_per_h"])
    X = split_xy(out, is_train=True)
    assert "event" not in X.columns and "time_to_hit_hours" not in X.columns


def test_c_index_bounds_and_direction():
    time = np.array([10.0, 20.0, 30.0, 40.0])
    event = np.array([1, 1, 1, 0])
    perfect = compute_c_index(time, event, risk=-time)
    inverted = compute_c_index(time, event, risk=time)
    assert perfect == 1.0
    assert inverted == 0.0
    assert compute_c_index(time, event, np.zeros(4)) == 0.5


def test_brier_excludes_rows_censored_before_horizon():
    time = np.array([10.0, 30.0, 5.0])
    event = np.array([1, 0, 0])          # third row censored before 24h
    prob = np.array([1.0, 0.0, 1.0])     # its wild prediction must be ignored
    assert compute_brier(time, event, prob, 24) == 0.0


def test_binary_target_masking():
    time = np.array([10.0, 30.0, 5.0])
    event = np.array([1, 0, 0])
    y, mask = make_binary_target(time, event, 24)
    assert list(y) == [1.0, 0.0, 0.0]
    assert list(mask) == [True, True, False]


def test_ipcw_weights_are_positive_and_finite():
    df = make_synthetic()
    w = compute_ipcw_weights(df["time_to_hit_hours"].values, df["event"].values, 48)
    assert np.isfinite(w).all() and (w > 0).all()


def test_monotonicity_repair():
    raw = np.array([[0.9, 0.2, 0.5, 1.2], [0.1, 0.1, 0.05, 0.3]])
    fixed = enforce_monotonicity(raw)
    assert (np.diff(fixed, axis=1) >= 0).all()
    assert fixed.max() <= 1.0 and fixed.min() >= 0.0


def test_hybrid_score_in_range():
    df = make_synthetic()
    p = np.clip(np.random.default_rng(1).uniform(size=(len(df), 3)), 0, 1)
    hybrid, c_idx, wb = compute_hybrid_score(
        df["time_to_hit_hours"].values, df["event"].values, p[:, 0], p[:, 1], p[:, 2]
    )
    assert 0 <= hybrid <= 1 and 0 <= c_idx <= 1 and 0 <= wb <= 1


def test_end_to_end_tiny_run():
    """Full path on a single config / two seeds — correctness, not accuracy."""
    from sksurv.util import Surv
    from src.models import GBSA_CONFIGS

    train = make_synthetic(n=120, seed=0)
    test = make_synthetic(n=40, seed=1).drop(columns=["event", "time_to_hit_hours"])
    sample = pd.DataFrame({"event_id": test["event_id"]})

    X_tr, X_te = split_xy(train, True), split_xy(test, False)
    X_tr_f = split_xy(create_features(train), True)
    X_te_f = split_xy(create_features(test), False)
    y = Surv.from_arrays(event=train["event"].astype(bool),
                         time=train["time_to_hit_hours"])

    tiny = [{**GBSA_CONFIGS[0], "n_estimators": 40}]
    gbsa_oof, gbsa_test = train_gbsa_ensemble(
        X_tr, y, X_te, train["event"].values, seeds=(0, 1),
        configs=tiny, n_folds=3, verbose=False,
    )
    lgb_oof, lgb_test = train_lgb_ipcw(
        X_tr_f, X_te_f, train["time_to_hit_hours"].values, train["event"].values,
        seeds=(0, 1), n_folds=3, verbose=False,
    )

    oof = blend(gbsa_oof, lgb_oof)
    assert oof.shape == (len(train), 4)
    assert (np.diff(oof, axis=1) >= -1e-12).all()

    sub = build_submission(test, blend(gbsa_test, lgb_test), sample)
    validate_submission(sub, sample)   # raises on any schema violation
    assert len(sub) == len(test)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
