"""
L2 - Chronological protocol and orchestration.

Protocol, applied identically to every series and every detector:

    1. build contextual features, dropping the rolling warm-up rows
    2. split at NAB's probationary boundary (first 15%, capped at 750 rows)
    3. FIT on the probationary slice only
    4. SCORE the remainder - the model never sees a scored row during fit
    5. threshold, group into incidents, evaluate

Two threshold calibrations are reported, because they answer different
questions:

    budget  threshold set on the scored region so every detector raises the
            same number of alerts. The fair comparison between methods - it
            removes alert volume as a confound.
    train   threshold set on the probationary region only. The honest
            deployment number: in practice you calibrate on known-good
            history and live with whatever volume arrives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import evaluation as ev
from .alerting import (
    flags_for_budget,
    flags_from_scores,
    group_incidents,
    incidents_to_frame,
    threshold_for_rate,
)
from .detectors import Detector, default_detectors
from .features import build_features
from .nab import Series, probation_end, window_bounds, window_mask

#: Alert budgets to report, as a fraction of scored observations.
TARGET_RATES = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
PRIMARY_RATE = 0.01
#: Quantile grid for the threshold sweep behind the PR / alert-cost curves.
SWEEP_RATES = tuple(np.round(np.geomspace(0.0005, 0.25, 36), 6))


@dataclass
class Prepared:
    """A series after feature construction and the chronological split."""

    series: Series
    featured: pd.DataFrame
    train: pd.DataFrame
    scored: pd.DataFrame
    bounds: list[tuple[int, int]]
    truth_mask: np.ndarray
    span_days: float
    period: pd.Timedelta


def prepare(series: Series) -> Prepared | None:
    """Apply steps 1-2 of the protocol. Returns None if the series is too short.

    Window bounds and the truth mask are recomputed *on the scored slice*, so
    every index handed to the evaluation layer refers to the same array. A
    window straddling the probationary boundary keeps only its scored tail,
    which is the correct treatment: the rest was never offered to the model.
    """
    featured = build_features(series.frame)
    if len(featured) < 60:
        return None

    split = probation_end(len(featured))
    if split < 20 or len(featured) - split < 40:
        return None

    train = featured.iloc[:split].reset_index(drop=True)
    scored = featured.iloc[split:].reset_index(drop=True)

    bounds = window_bounds(scored["timestamp"], series.windows)
    truth = window_mask(scored["timestamp"], series.windows)
    span = (scored["timestamp"].iloc[-1] - scored["timestamp"].iloc[0])
    period = series.period

    return Prepared(
        series=series, featured=featured, train=train, scored=scored,
        bounds=bounds, truth_mask=truth,
        span_days=span.total_seconds() / 86400.0, period=period,
    )


def _operating_point(prep: Prepared, detector: Detector, scores: np.ndarray,
                     flags: np.ndarray, threshold: float,
                     calibration: str, rate: float) -> dict:
    incidents = group_incidents(
        prep.scored["timestamp"], flags, scores, prep.scored["value"],
        period=prep.period,
    )
    row = {
        "key": prep.series.key,
        "corpus": prep.series.corpus,
        "name": prep.series.name,
        "detector": detector.name,
        "calibration": calibration,
        "target_rate": rate,
        "threshold": threshold,
        "n_scored": len(prep.scored),
        "n_flags": int(np.asarray(flags).sum()),
        "actual_rate": float(np.asarray(flags).mean()),
        "span_days": prep.span_days,
    }
    row.update(ev.evaluate(
        prep.scored["timestamp"], flags, incidents, prep.bounds,
        prep.truth_mask, prep.span_days,
    ))
    return row


def run_series(series: Series, detectors: list[Detector] | None = None,
               rates: tuple[float, ...] = TARGET_RATES,
               sweep_rates: tuple[float, ...] = SWEEP_RATES,
               primary_rate: float = PRIMARY_RATE):
    """Run the full protocol for one series across all detectors.

    Returns (metrics, sweep, scored_frame, incidents) where `scored_frame`
    carries one score column per detector for the dashboard to reuse without
    refitting.
    """
    prep = prepare(series)
    if prep is None:
        return [], [], None, []

    detectors = detectors or default_detectors()
    metrics: list[dict] = []
    sweep: list[dict] = []
    incident_rows: list[pd.DataFrame] = []

    export = prep.scored[["timestamp", "value", "rolling_mean_short",
                          "rolling_mean_long", "rolling_z"]].copy()
    export["in_window"] = prep.truth_mask

    for detector in detectors:
        detector.fit(prep.train)
        scores = detector.score(prep.scored)
        train_scores = detector.score(prep.train)
        export[f"score_{detector.name}"] = scores

        for rate in rates:
            # Budget: exact top-k by rank, so every detector is compared at
            # identical alert volume even where scores tie.
            metrics.append(_operating_point(
                prep, detector, scores, flags_for_budget(scores, rate),
                threshold_for_rate(scores, rate), "budget", rate))
            # Train: a real threshold learned from probationary history, with
            # whatever alert volume that produces on unseen data.
            threshold = threshold_for_rate(train_scores, rate)
            metrics.append(_operating_point(
                prep, detector, scores, flags_from_scores(scores, threshold),
                threshold, "train", rate))

        # Dense sweep for curves - budget calibration only, since the curve is
        # a property of the score ranking rather than of any one threshold.
        for rate in sweep_rates:
            threshold = threshold_for_rate(scores, rate)
            flags = flags_for_budget(scores, rate)
            incidents = group_incidents(
                prep.scored["timestamp"], flags, scores, prep.scored["value"],
                period=prep.period)
            point = ev.point_metrics(prep.truth_mask, flags)
            win = ev.window_metrics(prep.scored["timestamp"], flags, incidents,
                                    prep.bounds, prep.span_days)
            nab = ev.nab_score_components(
                ev.alert_flags(len(flags), incidents), prep.bounds, "standard")
            sweep.append({
                "key": series.key, "corpus": series.corpus,
                "detector": detector.name, "target_rate": rate,
                "threshold": threshold,
                "pa_precision": point["pa_precision"],
                "pa_recall": point["pa_recall"],
                "pa_f1": point["pa_f1"],
                "window_recall": win["window_recall"],
                "false_alarms_per_day": win["false_alarms_per_day"],
                "mttd_minutes": win["mttd_minutes"],
                "nab_raw_standard": nab["raw"],
                "nab_null_standard": nab["null"],
                "nab_perfect_standard": nab["perfect"],
            })

        # Incident queue exported at the headline operating point only.
        flags = flags_for_budget(scores, primary_rate)
        incidents = group_incidents(prep.scored["timestamp"], flags, scores,
                                    prep.scored["value"], period=prep.period)
        frame = incidents_to_frame(incidents)
        if not frame.empty:
            in_window = np.zeros(len(prep.scored), dtype=bool)
            for start, end in prep.bounds:
                in_window[start:end + 1] = True
            frame.insert(0, "key", series.key)
            frame.insert(1, "detector", detector.name)
            frame["hits_window"] = [
                bool(in_window[int(a):int(b) + 1].any())
                for a, b in zip(frame["start_idx"], frame["end_idx"])
            ]
            incident_rows.append(frame)

    return metrics, sweep, export, incident_rows


def aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread per detector / calibration / budget across series.

    Only labelled corpora contribute detection metrics; the unlabelled
    control corpus has no windows to detect and would otherwise drag every
    recall column to NaN.
    """
    labelled = metrics[metrics["n_windows"] > 0]
    grouped = labelled.groupby(["detector", "calibration", "target_rate"])
    summary = grouped.agg(
        n_series=("key", "nunique"),
        window_recall_mean=("window_recall", "mean"),
        window_recall_sd=("window_recall", "std"),
        pa_f1_mean=("pa_f1", "mean"),
        pa_f1_sd=("pa_f1", "std"),
        pa_precision_mean=("pa_precision", "mean"),
        pa_recall_mean=("pa_recall", "mean"),
        point_f1_mean=("point_f1", "mean"),
        point_recall_ceiling_mean=("point_recall_ceiling", "mean"),
        false_alarms_per_day_mean=("false_alarms_per_day", "mean"),
        false_alarms_per_day_sd=("false_alarms_per_day", "std"),
        alerts_per_day_mean=("alerts_per_day", "mean"),
        actual_rate_mean=("actual_rate", "mean"),
        mttd_minutes_mean=("mttd_minutes", "mean"),
        mttd_window_fraction_mean=("mttd_window_fraction", "mean"),
        incident_precision_mean=("incident_precision", "mean"),
    ).reset_index()

    # NAB normalises over the corpus, so sum the components first and
    # normalise once - averaging per-file normalised scores is not the same
    # number and would overweight short files.
    for suffix in ("standard", "low_FP_rate", "low_FN_rate"):
        totals = metrics.groupby(["detector", "calibration", "target_rate"]).agg(
            raw=(f"nab_raw_{suffix}", "sum"),
            null=(f"nab_null_{suffix}", "sum"),
            perfect=(f"nab_perfect_{suffix}", "sum"),
        ).reset_index()
        totals[f"nab_{suffix}"] = [
            ev.normalize(r, n, p)
            for r, n, p in zip(totals["raw"], totals["null"], totals["perfect"])
        ]
        summary = summary.merge(
            totals[["detector", "calibration", "target_rate", f"nab_{suffix}"]],
            on=["detector", "calibration", "target_rate"], how="left",
        )

    control = metrics[metrics["n_windows"] == 0]
    if not control.empty:
        control_summary = control.groupby(
            ["detector", "calibration", "target_rate"]
        ).agg(control_false_alarms_per_day=("false_alarms_per_day", "mean")).reset_index()
        summary = summary.merge(
            control_summary, on=["detector", "calibration", "target_rate"], how="left")

    return summary.sort_values(["calibration", "target_rate", "detector"])
