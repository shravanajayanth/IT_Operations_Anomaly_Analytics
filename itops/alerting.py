"""
L4 - From scores to reviewable alerts.

A detector that flags 40 scattered points has not produced 40 things to look
at; a sustained event lights up many consecutive rows. Grouping contiguous
flags into *incident candidates* is what turns a point-flag stream into a
triage queue, and it is the unit in which false-alarm cost should be counted:
an analyst pays once per incident, not once per sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .nab import as_index


@dataclass(frozen=True)
class Incident:
    """One contiguous run of flagged observations."""

    start_idx: int
    end_idx: int  # inclusive
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    peak_idx: int
    peak_time: pd.Timestamp
    peak_score: float
    peak_value: float
    n_points: int

    @property
    def duration(self) -> pd.Timedelta:
        return self.end_time - self.start_time


def threshold_for_rate(scores: np.ndarray, rate: float) -> float:
    """Smallest threshold that flags approximately `rate` of observations.

    Expressed as a quantile so that a target alert budget - rather than an
    arbitrary `contamination` guess - drives the operating point.
    """
    scores = np.asarray(scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return np.inf
    return float(np.quantile(finite, 1.0 - np.clip(rate, 0.0, 1.0)))


def flags_from_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Points strictly above `threshold`; non-finite scores never fire.

    Strict rather than inclusive because anomaly scores tie heavily in
    practice - a flatline series scores identically everywhere, and a forest
    fitted on a single feature produces large blocks of equal scores. With
    `>=`, a threshold that lands on such a block flags the entire block, so a
    nominal 1% budget can flag 100% of the series.
    """
    scores = np.asarray(scores, dtype=float)
    return np.where(np.isfinite(scores), scores > threshold, False)


def flags_for_budget(scores: np.ndarray, rate: float) -> np.ndarray:
    """Flag exactly the top `rate` fraction of observations.

    Thresholding cannot honour an alert budget when scores tie; ranking can.
    Ties are broken towards the earliest observation, which is both
    deterministic and the operationally sensible choice - given two equally
    anomalous points, the one you would rather have been told about is the
    one that happened first.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    budget = int(np.floor(np.clip(rate, 0.0, 1.0) * n))
    flags = np.zeros(n, dtype=bool)
    if budget == 0:
        return flags

    finite = np.isfinite(scores)
    if not finite.any():
        return flags
    # lexsort: primary key last. Sort by score descending, then index ascending.
    candidates = np.flatnonzero(finite)
    order = np.lexsort((candidates, -scores[candidates]))
    flags[candidates[order[:budget]]] = True
    return flags


def group_incidents(timestamps, flags, scores, values,
                    max_gap_periods: int = 3,
                    period: pd.Timedelta | None = None) -> list[Incident]:
    """Merge flagged points separated by at most `max_gap_periods` samples.

    The tolerance matters because detectors commonly flag the shoulders of an
    event but not its flat middle; without merging, one incident is counted as
    several, inflating both the alert count and the false-alarm rate.
    """
    ts = as_index(timestamps)
    flags = np.asarray(flags, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    values = np.asarray(values, dtype=float)

    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    if period is None:
        deltas = pd.Series(ts).diff().dropna()
        period = deltas.median() if not deltas.empty else pd.Timedelta(minutes=5)

    tolerance = period * max(1, max_gap_periods)
    incidents: list[Incident] = []
    run_start = idx[0]
    previous = idx[0]

    def close(start: int, end: int) -> Incident:
        span = np.arange(start, end + 1)
        inside = span[flags[span]]
        peak = int(inside[np.argmax(scores[inside])])
        return Incident(
            start_idx=int(start), end_idx=int(end),
            start_time=ts[start], end_time=ts[end],
            peak_idx=peak, peak_time=ts[peak],
            peak_score=float(scores[peak]), peak_value=float(values[peak]),
            n_points=int(inside.size),
        )

    for current in idx[1:]:
        # Gap measured in time, so collection gaps do not silently merge
        # events that are hours apart but adjacent in row order.
        if ts[current] - ts[previous] > tolerance:
            incidents.append(close(run_start, previous))
            run_start = current
        previous = current
    incidents.append(close(run_start, previous))
    return incidents


def incidents_to_frame(incidents: list[Incident]) -> pd.DataFrame:
    """Tabular form for `results/incidents.csv` and the dashboard queue."""
    if not incidents:
        return pd.DataFrame(columns=[
            "start_idx", "end_idx", "start_time", "end_time", "peak_idx",
            "peak_time", "peak_score", "peak_value", "n_points",
            "duration_minutes",
        ])
    rows = []
    for inc in incidents:
        rows.append({
            "start_idx": inc.start_idx, "end_idx": inc.end_idx,
            "start_time": inc.start_time, "end_time": inc.end_time,
            "peak_idx": inc.peak_idx, "peak_time": inc.peak_time,
            "peak_score": inc.peak_score, "peak_value": inc.peak_value,
            "n_points": inc.n_points,
            "duration_minutes": inc.duration.total_seconds() / 60.0,
        })
    return pd.DataFrame(rows)
