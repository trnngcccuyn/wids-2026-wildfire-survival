"""
Competition metrics and survival-analysis helpers.

The leaderboard metric is::

    Hybrid = 0.3 * C-index + 0.7 * (1 - WeightedBrier)
    WeightedBrier = 0.3 * Brier@24h + 0.4 * Brier@48h + 0.3 * Brier@72h

Brier is evaluated with censor-aware masking:

* a fire that hit by horizon H            -> label 1
* a fire censored *after* H (survived H)  -> label 0
* a fire censored *before* H              -> excluded (outcome unknown)

Every function here re-implements that rule exactly so that OOF numbers are
directly comparable to the leaderboard.
"""

from __future__ import annotations

import numpy as np

from src.config import BRIER_HORIZONS


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
def compute_c_index(
    time: np.ndarray, event: np.ndarray, risk: np.ndarray
) -> float:
    """
    Harrell's concordance index.

    A pair ``(i, j)`` is comparable when ``i`` experienced the event and hit
    strictly earlier than ``j`` was last observed. The pair is concordant when
    the model assigned ``i`` the higher risk; ties count as half credit.

    Vectorised over the full pair matrix — fine at n ~ 10^3, and identical in
    result to the naive double loop.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    risk = np.asarray(risk, dtype=float)

    comparable = (event[:, None] == 1) & (time[:, None] < time[None, :])
    n_comparable = comparable.sum()
    if n_comparable == 0:
        return 0.5

    higher = (risk[:, None] > risk[None, :]) & comparable
    tied = (risk[:, None] == risk[None, :]) & comparable
    return float((higher.sum() + 0.5 * tied.sum()) / n_comparable)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
def compute_brier(
    time: np.ndarray, event: np.ndarray, prob: np.ndarray, horizon: float
) -> float:
    """Censor-aware Brier score at ``horizon`` (lower is better)."""
    time = np.asarray(time, dtype=float)
    event = np.asarray(event)
    prob = np.asarray(prob, dtype=float)

    # Rows censored before the horizon have an unknown outcome -> excluded.
    valid = ~((event == 0) & (time < horizon))
    if valid.sum() == 0:
        return 0.25  # uninformative baseline

    y_true = ((event == 1) & (time <= horizon)).astype(float)[valid]
    return float(np.mean((np.clip(prob[valid], 0, 1) - y_true) ** 2))


def compute_hybrid_score(
    time: np.ndarray,
    event: np.ndarray,
    p24: np.ndarray,
    p48: np.ndarray,
    p72: np.ndarray,
) -> tuple[float, float, float]:
    """
    Reproduce the leaderboard metric.

    The risk score fed to the C-index reuses the Brier horizon weights, so a
    single ranking is optimised jointly with calibration rather than separately.

    Returns
    -------
    (hybrid, c_index, weighted_brier)
    """
    risk = 0.3 * np.asarray(p24) + 0.4 * np.asarray(p48) + 0.3 * np.asarray(p72)
    c_idx = compute_c_index(time, event, risk)

    briers = {
        24: compute_brier(time, event, p24, 24),
        48: compute_brier(time, event, p48, 48),
        72: compute_brier(time, event, p72, 72),
    }
    weighted_brier = sum(BRIER_HORIZONS[h] * briers[h] for h in BRIER_HORIZONS)
    hybrid = 0.3 * c_idx + 0.7 * (1 - weighted_brier)
    return float(hybrid), float(c_idx), float(weighted_brier)


# --------------------------------------------------------------------------- #
# Survival <-> binary bridge
# --------------------------------------------------------------------------- #
def make_binary_target(
    time_vals: np.ndarray, event_vals: np.ndarray, horizon: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Turn the survival target into a binary classification target at ``horizon``.

    Returns
    -------
    (y, mask)
        ``y`` is 1 for fires that hit by the horizon. ``mask`` is False for rows
        censored before the horizon, whose label is genuinely unknown and which
        must be dropped from training and evaluation alike.
    """
    time_vals = np.asarray(time_vals, dtype=float)
    event_vals = np.asarray(event_vals)
    unknown = (event_vals == 0) & (time_vals < horizon)
    y = ((event_vals == 1) & (time_vals <= horizon)).astype(float)
    return y, ~unknown


def compute_ipcw_weights(
    times: np.ndarray, events: np.ndarray, horizon: float
) -> np.ndarray:
    """
    Inverse-probability-of-censoring weights.

    Dropping censored rows biases the binary classifier: fires that vanish from
    observation early are not a random sample. IPCW corrects for this by
    up-weighting each retained row by ``1 / G(t)``, where ``G`` is the
    Kaplan-Meier estimate of the *censoring* survival function.

    ``G`` is floored at 0.01 to keep weights from exploding in the tail.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events)

    unique_t = np.sort(np.unique(times))
    surv = np.ones(len(unique_t))
    for i, t in enumerate(unique_t):
        at_risk = (times >= t).sum()
        censored_at_t = ((times == t) & (events == 0)).sum()
        if at_risk > 0:
            surv[i] = 1 - censored_at_t / at_risk
        if i > 0:
            surv[i] *= surv[i - 1]

    def g(t: float) -> float:
        idx = np.searchsorted(unique_t, t, side="right") - 1
        return max(surv[idx], 0.01) if idx >= 0 else 1.0

    weights = np.ones(len(times))
    for i in range(len(times)):
        if events[i] == 1 and times[i] <= horizon:
            weights[i] = 1.0 / g(times[i])
        elif times[i] >= horizon:
            weights[i] = 1.0 / g(horizon)
    return weights


# --------------------------------------------------------------------------- #
# Submission constraints
# --------------------------------------------------------------------------- #
def enforce_monotonicity(preds: np.ndarray) -> np.ndarray:
    """
    Clip to [0, 1] and enforce ``p12 <= p24 <= p48 <= p72`` row-wise.

    This is a hard submission requirement, and it is also correct: a cumulative
    hit probability cannot decrease as the horizon extends. Blending two models
    per column can break it, so we repair with a running maximum.
    """
    result = np.clip(np.asarray(preds, dtype=float), 0, 1)
    for i in range(1, result.shape[1]):
        result[:, i] = np.maximum(result[:, i], result[:, i - 1])
    return result


def validate_submission(sub, sample_sub) -> None:
    """Raise if the submission would be rejected by the competition validator."""
    required = ["event_id", "prob_12h", "prob_24h", "prob_48h", "prob_72h"]
    assert list(sub.columns) == required, f"Bad schema: {list(sub.columns)}"
    assert len(sub) == len(sample_sub), "Row count differs from sample submission"
    assert sub["event_id"].is_unique, "Duplicate event_id"
    assert set(sub["event_id"]) == set(sample_sub["event_id"]), "event_id mismatch"
    assert sub[required[1:]].notna().all().all(), "NaN probability"

    probs = sub[required[1:]].to_numpy()
    assert probs.min() >= 0 and probs.max() <= 1, "Probability outside [0, 1]"
    assert (np.diff(probs, axis=1) >= -1e-12).all(), "Monotonicity violated"
