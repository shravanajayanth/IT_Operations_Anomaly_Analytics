"""
L5 - Evaluation at three levels.

NAB labels are time windows covering 10% of each file, so comparing a
detector that flags ~1% of points against the expanded window mask caps
recall at ~0.10 and F1 at ~0.18 *regardless of how good the detector is*.
`recall_ceiling` computes that bound explicitly, and `point_metrics` reports
the raw point-wise numbers next to it so the distortion is visible rather
than silently reported as poor performance.

The three levels answer three different questions:

    window          did we catch the incident at all?
    point-adjusted  how well did we cover it once caught?
    operational     what did catching it cost in false alarms and delay?

plus `nab_score_components`, a faithful reimplementation of NAB's own scorer
as an external yardstick.

References:
    Lavin & Ahmad (2015), IEEE ICMLA - NAB scoring and application profiles.
    Xu et al. (2018), WWW - point-adjustment protocol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .nab import as_index

#: NAB application profiles. `fp` is the penalty for a false positive far
#: from any window; `fn` the penalty for missing a window entirely. The
#: low-FP profile doubles the cost of noise, the low-FN profile doubles the
#: cost of a miss - which bracket is right depends on how expensive an
#: analyst's attention is relative to an undetected incident.
COST_PROFILES = {
    "standard": {"tp": 1.0, "fp": 0.11, "fn": 1.0},
    "reward_low_FP_rate": {"tp": 1.0, "fp": 0.22, "fn": 1.0},
    "reward_low_FN_rate": {"tp": 1.0, "fp": 0.11, "fn": 2.0},
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def scaled_sigmoid(relative_position: float) -> float:
    """NAB's reward curve over position relative to a window.

    `relative_position` is measured in window lengths from the window *end*:
    -1.0 at the window start, 0.0 at its end, positive after it. Early
    detections score near +1, a detection at the closing edge scores 0, and
    anything beyond three window lengths away scores the full -1.
    """
    if relative_position > 3.0:
        return -1.0
    return 2.0 * _sigmoid(-5.0 * relative_position) - 1.0


#: Normalising constant so a detection on the first row of a window earns
#: exactly `tp` weight.
_PERFECT_TP = scaled_sigmoid(-1.0)


# --------------------------------------------------------------------------
# Point level
# --------------------------------------------------------------------------

def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive ``(start, end)`` bounds of each contiguous True run."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate(([False], mask, [False]))
    change = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def point_adjust(truth_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Expand predictions to whole truth segments that were hit at all.

    The standard protocol for segment-labelled series: an operator alerted
    anywhere inside an event has detected that event, so crediting only the
    exact rows flagged understates detection of extended incidents.
    """
    truth_mask = np.asarray(truth_mask, dtype=bool)
    adjusted = np.asarray(pred_mask, dtype=bool).copy()
    for start, end in segments(truth_mask):
        if adjusted[start:end + 1].any():
            adjusted[start:end + 1] = True
    return adjusted


def recall_ceiling(truth_mask: np.ndarray, n_flags: int) -> float:
    """Highest point-wise recall reachable with `n_flags` alerts.

    Reported alongside raw recall to show how much of an apparently poor
    score is structural rather than a property of the detector.
    """
    n_truth = int(np.asarray(truth_mask, dtype=bool).sum())
    if n_truth == 0:
        return float("nan")
    return min(1.0, n_flags / n_truth)


def _prf(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, dict]:
    truth = np.asarray(truth, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    tn = int((~truth & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def point_metrics(truth_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    """Raw and point-adjusted precision/recall/F1 over the window mask."""
    raw_p, raw_r, raw_f1, raw_cm = _prf(truth_mask, pred_mask)
    adjusted = point_adjust(truth_mask, pred_mask)
    adj_p, adj_r, adj_f1, adj_cm = _prf(truth_mask, adjusted)
    n_flags = int(np.asarray(pred_mask, dtype=bool).sum())
    return {
        "point_precision": raw_p,
        "point_recall": raw_r,
        "point_f1": raw_f1,
        "point_tp": raw_cm["tp"], "point_fp": raw_cm["fp"],
        "point_fn": raw_cm["fn"], "point_tn": raw_cm["tn"],
        "point_recall_ceiling": recall_ceiling(truth_mask, n_flags),
        "pa_precision": adj_p,
        "pa_recall": adj_r,
        "pa_f1": adj_f1,
        "pa_tp": adj_cm["tp"], "pa_fp": adj_cm["fp"], "pa_fn": adj_cm["fn"],
    }


# --------------------------------------------------------------------------
# Window and operational level
# --------------------------------------------------------------------------

def window_metrics(timestamps, flags, incidents, bounds, span_days: float) -> dict:
    """Did we catch each window, how fast, and at what alert cost?

    `bounds` are inclusive index pairs within the *scored* region. False
    alarms are counted per grouped incident rather than per flagged point,
    because that is the unit of analyst effort.
    """
    ts = as_index(timestamps)
    flags = np.asarray(flags, dtype=bool)
    flagged = np.flatnonzero(flags)

    detected = 0
    delays: list[float] = []
    delay_fractions: list[float] = []
    for start, end in bounds:
        hits = flagged[(flagged >= start) & (flagged <= end)]
        if hits.size == 0:
            continue
        detected += 1
        first = int(hits[0])
        # Delay is measured from the window start. NAB windows open before
        # the labelled anomaly, so this is an upper bound on true latency.
        delay = (ts[first] - ts[start]).total_seconds()
        window_span = max((ts[end] - ts[start]).total_seconds(), 1e-9)
        delays.append(delay)
        delay_fractions.append(delay / window_span)

    in_window = np.zeros(len(ts), dtype=bool)
    for start, end in bounds:
        in_window[start:end + 1] = True

    true_incidents = sum(
        1 for inc in incidents
        if in_window[inc.start_idx:inc.end_idx + 1].any()
    )
    n_incidents = len(incidents)
    false_incidents = n_incidents - true_incidents

    return {
        "n_windows": len(bounds),
        "n_windows_detected": detected,
        "window_recall": detected / len(bounds) if bounds else float("nan"),
        "n_incidents": n_incidents,
        "n_true_incidents": true_incidents,
        "n_false_incidents": false_incidents,
        "incident_precision": (true_incidents / n_incidents) if n_incidents else float("nan"),
        "false_alarms_per_day": (false_incidents / span_days) if span_days > 0 else float("nan"),
        "alerts_per_day": (n_incidents / span_days) if span_days > 0 else float("nan"),
        "mttd_minutes": float(np.mean(delays) / 60.0) if delays else float("nan"),
        "mttd_window_fraction": float(np.mean(delay_fractions)) if delay_fractions else float("nan"),
    }


# --------------------------------------------------------------------------
# NAB score
# --------------------------------------------------------------------------

def nab_score_components(flags, bounds, profile: str = "standard") -> dict:
    """Unnormalised NAB score plus its null and perfect references.

    Returned un-normalised because NAB normalises over the whole corpus, not
    per file: a file with no labelled windows has a perfect score of 0 and
    cannot be normalised alone, but can still contribute false-positive
    penalties to the corpus total. `normalize` does the final step.

    Scoring rules: only the *earliest* detection inside a window is rewarded
    (later ones inside the same window are ignored, neither paid nor
    penalised); a missed window costs `fn`; a detection outside every window
    costs up to `fp`, discounted if it falls shortly after a window, since
    such a flag may be a late detection of the same event.
    """
    if profile not in COST_PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(COST_PROFILES)}")
    weights = COST_PROFILES[profile]
    flags = np.asarray(flags, dtype=bool)
    flagged = np.flatnonzero(flags)

    in_window = np.zeros(len(flags), dtype=bool)
    for start, end in bounds:
        in_window[start:end + 1] = True

    raw = 0.0
    n_tp = n_fn = 0
    for start, end in bounds:
        hits = flagged[(flagged >= start) & (flagged <= end)]
        if hits.size == 0:
            raw -= weights["fn"]
            n_fn += 1
            continue
        size = end - start + 1
        position = -float(end - int(hits[0]) + 1) / size
        raw += scaled_sigmoid(position) * weights["tp"] / _PERFECT_TP
        n_tp += 1

    ordered = sorted(bounds, key=lambda b: b[1])
    false_positives = flagged[~in_window[flagged]] if flagged.size else np.array([], dtype=int)
    for idx in false_positives:
        preceding = [b for b in ordered if b[1] < idx]
        if not preceding:
            raw -= weights["fp"]
            continue
        start, end = preceding[-1]
        size = max(end - start + 1, 2)
        position = abs(int(idx) - end) / float(size - 1)
        raw += scaled_sigmoid(position) * weights["fp"]

    return {
        "profile": profile,
        "raw": float(raw),
        "null": float(-weights["fn"] * len(bounds)),
        "perfect": float(weights["tp"] * len(bounds)),
        "nab_tp": n_tp,
        "nab_fn": n_fn,
        "nab_fp": int(false_positives.size),
    }


def alert_flags(n_rows: int, incidents) -> np.ndarray:
    """Reduce each incident to the single row on which the alert fired.

    NAB's reference scorer charges a false-positive penalty per flagged
    *row*, which assumes one row equals one alert. This project groups
    contiguous flags into incidents precisely because that assumption is
    false for sustained events: a twenty-row excursion is one thing to
    investigate, not twenty. Scoring the incident's first row keeps one
    penalty per alert and preserves NAB's reward for detecting early.
    """
    flags = np.zeros(n_rows, dtype=bool)
    for incident in incidents:
        flags[incident.start_idx] = True
    return flags


def normalize(raw: float, null: float, perfect: float) -> float:
    """Map a raw NAB score onto 0 (no detections) .. 100 (perfect)."""
    denominator = perfect - null
    if denominator == 0:
        return float("nan")
    return 100.0 * (raw - null) / denominator


def evaluate(timestamps, flags, incidents, bounds, truth_mask,
             span_days: float) -> dict:
    """All three levels for one (series, detector, operating point)."""
    result: dict = {}
    result.update(point_metrics(truth_mask, flags))
    result.update(window_metrics(timestamps, flags, incidents, bounds, span_days))

    # Primary: one alert per incident (see `alert_flags`).
    alerts = alert_flags(len(np.asarray(flags)), incidents)
    for profile in COST_PROFILES:
        components = nab_score_components(alerts, bounds, profile)
        suffix = profile.replace("reward_", "")
        result[f"nab_raw_{suffix}"] = components["raw"]
        result[f"nab_null_{suffix}"] = components["null"]
        result[f"nab_perfect_{suffix}"] = components["perfect"]
        result[f"nab_norm_{suffix}"] = normalize(
            components["raw"], components["null"], components["perfect"]
        )
    result["nab_tp"] = components["nab_tp"]
    result["nab_fp"] = components["nab_fp"]
    result["nab_fn"] = components["nab_fn"]

    # Secondary: NAB's own per-row convention, retained so the numbers remain
    # comparable with published NAB leaderboard entries.
    pointwise = nab_score_components(flags, bounds, "standard")
    result["nab_raw_pointwise"] = pointwise["raw"]
    result["nab_null_pointwise"] = pointwise["null"]
    result["nab_perfect_pointwise"] = pointwise["perfect"]
    result["nab_fp_pointwise"] = pointwise["nab_fp"]
    return result
