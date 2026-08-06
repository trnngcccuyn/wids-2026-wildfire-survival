"""
Feature engineering.

All features are derived exclusively from the first five hours after the
initial perimeter observation (t0), matching the competition's information
constraint. Nothing here uses the target or any post-t0+5h signal.

Design notes
------------
The raw columns describe three physical quantities that drive whether a fire
reaches an evacuation zone:

1. **Proximity**  - how far the perimeter already is from the nearest zone.
2. **Kinematics** - how fast it is closing that gap (translation + radial growth).
3. **Intensity**  - how much fuel it is consuming per hour and how erratic the
   perimeter observations are.

The engineered block below mostly expresses interactions between those three,
plus an explicit "time to contact" estimate (``eta_effective``) which is the
closest thing to a physical prior for a survival target measured in hours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Identifier and target columns that must never be fed to a model.
ID_COL = "event_id"
TARGET_COLS = ["event", "time_to_hit_hours"]

# The ten signals that survive aggressive ablation. Kept as a reference list
# for ablation studies and for interpreting feature importance; the production
# models use the wider engineered frame.
CORE_FEATURES = [
    "dist_min_ci_0_5h",
    "dist_km",
    "log_distance",
    "closing_speed_m_per_h",
    "radial_growth_rate_m_per_h",
    "num_perimeters_0_5h",
    "area_first_ha",
    "area_growth_rate_ha_per_h",
    "alignment_abs",
    "eta_effective",
]

# Columns dropped because they are noisy, near-duplicate, or unstable across
# folds on a 221-row training set.
DROP_COLS = [
    "relative_growth_0_5h",
    "projected_advance_m",
    "centroid_displacement_m",
    "centroid_speed_m_per_h",
    "closing_speed_abs_m_per_h",
    "area_growth_abs_0_5h",
]

# Sentinel used when a fire is not closing at all: an "infinite" ETA.
ETA_SENTINEL = 9999.0


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the engineered feature frame used by the LightGBM IPCW models.

    Parameters
    ----------
    df
        Raw train or test frame as shipped by the competition.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with engineered columns added, noisy columns dropped and
        all non-finite values replaced by 0.
    """
    r = df.copy()

    dist = r["dist_min_ci_0_5h"].clip(lower=1)  # metres; avoid divide-by-zero
    speed = r["closing_speed_m_per_h"]
    perimeters = r["num_perimeters_0_5h"]
    area_first = r["area_first_ha"]

    # -- Distance transforms -------------------------------------------------
    # Risk falls off sharply and non-linearly with distance, so we hand the
    # trees several monotone re-scalings and let them pick the useful split.
    r["log_distance"] = np.log1p(dist)
    r["inv_distance"] = 1 / (dist / 1000 + 0.1)
    r["inv_distance_sq"] = r["inv_distance"] ** 2
    r["sqrt_distance"] = np.sqrt(dist)
    r["dist_km"] = dist / 1000
    r["dist_km_sq"] = (dist / 1000) ** 2
    r["dist_km_cb"] = (dist / 1000) ** 3
    r["dist_rank"] = dist.rank(pct=True)

    # -- Area relative to distance ------------------------------------------
    # A 500 ha fire 2 km away is a different problem from a 5 ha fire 2 km away.
    # Convert burned area into an equivalent circular radius so the two
    # quantities share units.
    fire_radius = np.sqrt(area_first * 10000 / np.pi)  # ha -> m^2 -> radius (m)
    r["fire_radius_km"] = fire_radius / 1000
    r["radius_to_dist"] = fire_radius / dist
    r["area_to_dist_ratio"] = area_first / (dist / 1000 + 0.1)
    r["log_area_dist_ratio"] = np.log1p(area_first) - np.log1p(dist)

    # -- Kinematics ----------------------------------------------------------
    # A perimeter can close on a zone two ways: the whole fire translates, or it
    # expands radially. Summing both gives a far better ETA than translation
    # alone, which is why `eta_effective` outperforms `eta_hours`.
    r["has_movement"] = (perimeters > 1).astype(float)
    closing_pos = speed.clip(lower=0)
    r["eta_hours"] = np.where(
        closing_pos > 0.01, dist / closing_pos, ETA_SENTINEL
    ).clip(max=ETA_SENTINEL)
    r["log_eta"] = np.log1p(r["eta_hours"].clip(0, ETA_SENTINEL))

    radial_growth = r["radial_growth_rate_m_per_h"].clip(lower=0)
    effective_closing = closing_pos + radial_growth
    r["effective_closing_speed"] = effective_closing
    r["eta_effective"] = np.where(
        effective_closing > 0.01, dist / effective_closing, ETA_SENTINEL
    ).clip(max=ETA_SENTINEL)

    # -- Composite threat scores --------------------------------------------
    # `alignment_abs` measures whether the spread direction points at the zone;
    # speed only matters if it is pointed the right way.
    r["threat_score"] = r["alignment_abs"] * speed / np.log1p(dist)
    r["threat_score_sq"] = r["threat_score"] ** 2
    r["fire_urgency"] = perimeters * speed
    r["growth_intensity"] = r["area_growth_rate_ha_per_h"] * perimeters

    # -- Operational distance bands -----------------------------------------
    # Mirrors how incident commanders actually bucket proximity.
    r["zone_critical"] = (dist < 5000).astype(float)
    r["zone_warning"] = ((dist >= 5000) & (dist < 10000)).astype(float)
    r["zone_safe"] = (dist >= 10000).astype(float)

    # -- Temporal context ----------------------------------------------------
    r["is_summer"] = r["event_start_month"].isin([6, 7, 8]).astype(float)
    r["is_afternoon"] = (
        (r["event_start_hour"] >= 12) & (r["event_start_hour"] < 20)
    ).astype(float)

    # -- Cleanup -------------------------------------------------------------
    r = r.drop(columns=[c for c in DROP_COLS if c in r.columns])
    r = r.replace([np.inf, -np.inf], np.nan).fillna(0)
    return r


def split_xy(
    df: pd.DataFrame, is_train: bool = True
) -> pd.DataFrame:
    """Drop identifier and target columns, returning a model-ready matrix."""
    drop = [ID_COL] + (TARGET_COLS if is_train else [])
    return df.drop(columns=[c for c in drop if c in df.columns])


def feature_names(df: pd.DataFrame) -> list[str]:
    """Names of the columns a model will actually see."""
    return [c for c in df.columns if c not in [ID_COL] + TARGET_COLS]
