"""
L1 - Contextual feature construction.

An Isolation Forest fitted on the raw metric alone can only isolate globally
extreme values, which is exactly what a static threshold already does. The
features below add *temporal context*, so that a sustained shift at an
otherwise unremarkable absolute level becomes isolable - the failure mode
static thresholds miss.

Every rolling statistic is shifted by one observation, so a row is only ever
compared against strictly prior history. Without the shift the current value
contributes to its own baseline and the residual is damped toward zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .nab import infer_period

#: Clip for the rolling z-score. Flat stretches drive the rolling standard
#: deviation to ~0, which would otherwise produce unbounded scores that
#: dominate the forest's split ranges.
Z_CLIP = 50.0
STD_FLOOR = 1e-9

#: Columns consumed by the contextual detector, in a fixed order so that
#: feature-attribution plots stay comparable across runs.
CONTEXT_FEATURES = [
    "value",
    "residual_short",
    "residual_long",
    "rolling_z",
    "rolling_std_short",
    "diff_1",
    "pct_change",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

RAW_FEATURES = ["value"]
BASELINE_FEATURES = ["rolling_z"]


def window_sizes(timestamps: pd.Series) -> tuple[int, int]:
    """Rolling window lengths in rows, for roughly one hour and one day.

    Expressed in rows rather than fixed constants because NAB files are
    sampled at anything from one to sixty minutes.
    """
    period = infer_period(timestamps)
    per_hour = max(3, int(np.ceil(pd.Timedelta(hours=1) / period)))
    per_day = max(per_hour * 2, int(np.ceil(pd.Timedelta(days=1) / period)))
    return per_hour, per_day


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return `frame` with contextual features appended and warm-up dropped.

    The leading rows where the long rolling window is not yet populated are
    removed rather than imputed: filling them would invent a baseline that
    the detector would then treat as observed history.
    """
    out = frame.copy()
    value = out["value"].astype(float)
    short, long = window_sizes(out["timestamp"])

    # `.shift(1)` => statistics over the preceding `short`/`long` rows only.
    roll_mean_short = value.rolling(short, min_periods=short).mean().shift(1)
    roll_std_short = value.rolling(short, min_periods=short).std(ddof=0).shift(1)
    roll_mean_long = value.rolling(long, min_periods=long).mean().shift(1)

    out["rolling_mean_short"] = roll_mean_short
    out["rolling_std_short"] = roll_std_short
    out["rolling_mean_long"] = roll_mean_long

    # Residuals are what make a contextual shift visible to a tree split.
    out["residual_short"] = value - roll_mean_short
    out["residual_long"] = value - roll_mean_long
    out["rolling_z"] = (
        out["residual_short"] / roll_std_short.clip(lower=STD_FLOOR)
    ).clip(-Z_CLIP, Z_CLIP)

    out["diff_1"] = value.diff()
    out["pct_change"] = value.pct_change().replace([np.inf, -np.inf], np.nan)
    out["lag_1"] = value.shift(1)

    # Calendar position as sine/cosine pairs so that 23:55 and 00:05 are
    # adjacent rather than maximally distant.
    hour = out["timestamp"].dt.hour + out["timestamp"].dt.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = out["timestamp"].dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    required = sorted(set(CONTEXT_FEATURES) | {"lag_1"})
    return out.dropna(subset=required).reset_index(drop=True)
