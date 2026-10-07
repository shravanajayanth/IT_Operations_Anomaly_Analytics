"""Correctness tests for the scoring layer.

The NAB score is a reimplementation, so it is pinned against the properties
its definition guarantees: a silent detector scores 0, a detector firing on
the first row of every window scores 100, and only the earliest hit inside a
window is ever paid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itops import evaluation as ev
from itops.alerting import (
    flags_for_budget,
    flags_from_scores,
    group_incidents,
    threshold_for_rate,
)
from itops.features import build_features


def flags_at(n: int, indices) -> np.ndarray:
    flags = np.zeros(n, dtype=bool)
    flags[list(indices)] = True
    return flags


# --------------------------------------------------------------------------
# scaled sigmoid
# --------------------------------------------------------------------------

def test_scaled_sigmoid_window_start_is_near_one():
    assert ev.scaled_sigmoid(-1.0) == pytest.approx(0.98661, abs=1e-5)


def test_scaled_sigmoid_window_end_is_zero():
    assert ev.scaled_sigmoid(0.0) == pytest.approx(0.0, abs=1e-12)


def test_scaled_sigmoid_saturates_beyond_three_windows():
    assert ev.scaled_sigmoid(3.5) == -1.0
    assert ev.scaled_sigmoid(100.0) == -1.0


def test_scaled_sigmoid_is_monotone_decreasing():
    xs = np.linspace(-2.0, 3.0, 50)
    ys = [ev.scaled_sigmoid(x) for x in xs]
    assert all(b <= a + 1e-12 for a, b in zip(ys, ys[1:]))


# --------------------------------------------------------------------------
# NAB score anchors
# --------------------------------------------------------------------------

def test_silent_detector_scores_null():
    bounds = [(10, 20), (50, 60)]
    out = ev.nab_score_components(np.zeros(100, dtype=bool), bounds)
    assert out["raw"] == pytest.approx(out["null"])
    assert ev.normalize(out["raw"], out["null"], out["perfect"]) == pytest.approx(0.0)
    assert out["nab_fn"] == 2 and out["nab_tp"] == 0


def test_detection_on_first_row_of_every_window_scores_perfect():
    bounds = [(10, 20), (50, 60)]
    out = ev.nab_score_components(flags_at(100, [10, 50]), bounds)
    assert out["raw"] == pytest.approx(out["perfect"])
    assert ev.normalize(out["raw"], out["null"], out["perfect"]) == pytest.approx(100.0)


def test_only_earliest_hit_in_a_window_is_paid():
    bounds = [(10, 20)]
    once = ev.nab_score_components(flags_at(100, [10]), bounds)["raw"]
    repeatedly = ev.nab_score_components(flags_at(100, [10, 12, 15, 20]), bounds)["raw"]
    assert once == pytest.approx(repeatedly)


def test_later_detection_in_window_scores_less_than_earlier():
    bounds = [(10, 30)]
    early = ev.nab_score_components(flags_at(100, [11]), bounds)["raw"]
    late = ev.nab_score_components(flags_at(100, [29]), bounds)["raw"]
    assert early > late


def test_false_positive_before_any_window_costs_full_weight():
    bounds = [(50, 60)]
    out = ev.nab_score_components(flags_at(100, [5, 50]), bounds)
    expected = out["perfect"] - ev.COST_PROFILES["standard"]["fp"]
    assert out["raw"] == pytest.approx(expected)
    assert out["nab_fp"] == 1


def test_false_positive_just_after_a_window_is_discounted():
    bounds = [(10, 30)]
    near = ev.nab_score_components(flags_at(200, [10, 32]), bounds)["raw"]
    far = ev.nab_score_components(flags_at(200, [10, 199]), bounds)["raw"]
    assert near > far


def test_low_fp_profile_penalises_noise_harder():
    bounds = [(10, 20)]
    flags = flags_at(300, [10, 200, 250])
    standard = ev.nab_score_components(flags, bounds, "standard")["raw"]
    low_fp = ev.nab_score_components(flags, bounds, "reward_low_FP_rate")["raw"]
    assert low_fp < standard


def test_low_fn_profile_penalises_misses_harder():
    bounds = [(10, 20)]
    silent = np.zeros(100, dtype=bool)
    standard = ev.nab_score_components(silent, bounds, "standard")["raw"]
    low_fn = ev.nab_score_components(silent, bounds, "reward_low_FN_rate")["raw"]
    assert low_fn < standard


def test_unknown_profile_rejected():
    with pytest.raises(ValueError):
        ev.nab_score_components(np.zeros(10, dtype=bool), [(1, 2)], "nonsense")


def test_file_without_windows_has_undefined_normalised_score():
    out = ev.nab_score_components(flags_at(100, [5, 6]), [])
    assert out["perfect"] == 0.0 and out["null"] == 0.0
    assert np.isnan(ev.normalize(out["raw"], out["null"], out["perfect"]))
    # ...but still contributes its false-positive penalty to a corpus total.
    assert out["raw"] < 0


# --------------------------------------------------------------------------
# point level
# --------------------------------------------------------------------------

def test_alert_flags_charges_one_penalty_per_incident():
    """A sustained excursion is one alert, not one per row."""
    n = 200
    ts = _clock(n)
    flags = flags_at(n, range(100, 120))  # one 20-row excursion, no windows
    incidents = group_incidents(ts, flags, np.ones(n), np.ones(n))
    assert len(incidents) == 1

    per_row = ev.nab_score_components(flags, [(5, 10)])
    per_alert = ev.nab_score_components(ev.alert_flags(n, incidents), [(5, 10)])
    assert per_row["nab_fp"] == 20
    assert per_alert["nab_fp"] == 1
    assert per_alert["raw"] > per_row["raw"]


def test_alert_flags_fires_on_the_first_row_preserving_early_reward():
    n = 100
    ts = _clock(n)
    flags = flags_at(n, [40, 41, 42])
    incidents = group_incidents(ts, flags, np.ones(n), np.ones(n))
    assert list(np.flatnonzero(ev.alert_flags(n, incidents))) == [40]


def test_segments_finds_contiguous_runs():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert ev.segments(mask) == [(1, 2), (5, 5), (7, 9)]


def test_segments_handles_edges():
    assert ev.segments(np.array([1, 1, 0, 1], dtype=bool)) == [(0, 1), (3, 3)]
    assert ev.segments(np.zeros(5, dtype=bool)) == []


def test_point_adjust_expands_a_hit_segment():
    truth = np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=bool)
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    adjusted = ev.point_adjust(truth, pred)
    assert adjusted[1:4].all()
    assert not adjusted[5:7].any()  # untouched segment stays undetected


def test_point_adjust_never_invents_outside_truth():
    truth = np.array([0, 1, 1, 0], dtype=bool)
    pred = np.array([1, 0, 0, 1], dtype=bool)
    assert list(ev.point_adjust(truth, pred)) == [True, False, False, True]


def test_recall_ceiling_exposes_the_structural_cap():
    truth = np.zeros(1000, dtype=bool)
    truth[:100] = True  # 10% window coverage, as NAB guarantees
    assert ev.recall_ceiling(truth, n_flags=10) == pytest.approx(0.10)


def test_point_metrics_reports_ceiling_below_recall_is_impossible():
    truth = np.zeros(1000, dtype=bool)
    truth[:100] = True
    pred = np.zeros(1000, dtype=bool)
    pred[:10] = True
    out = ev.point_metrics(truth, pred)
    assert out["point_recall"] <= out["point_recall_ceiling"] + 1e-12
    assert out["pa_recall"] == pytest.approx(1.0)  # whole window credited


# --------------------------------------------------------------------------
# window / operational level
# --------------------------------------------------------------------------

def _clock(n: int, minutes: int = 5) -> pd.Series:
    return pd.Series(pd.date_range("2024-01-01", periods=n, freq=f"{minutes}min"))


def test_window_metrics_counts_detection_and_delay():
    n = 300
    ts = _clock(n)
    bounds = [(100, 140), (200, 240)]
    flags = flags_at(n, [110, 111])  # first window only, 10 samples late
    incidents = group_incidents(ts, flags, np.ones(n), np.ones(n))
    out = ev.window_metrics(ts, flags, incidents, bounds, span_days=1.0)
    assert out["n_windows"] == 2
    assert out["n_windows_detected"] == 1
    assert out["window_recall"] == pytest.approx(0.5)
    assert out["mttd_minutes"] == pytest.approx(50.0)
    assert out["n_false_incidents"] == 0


def test_false_alarms_counted_per_incident_not_per_point():
    n = 300
    ts = _clock(n)
    bounds = [(100, 140)]
    # Ten consecutive flagged points far from the window = ONE false alarm.
    flags = flags_at(n, range(10, 20))
    incidents = group_incidents(ts, flags, np.ones(n), np.ones(n))
    out = ev.window_metrics(ts, flags, incidents, bounds, span_days=1.0)
    assert out["n_incidents"] == 1
    assert out["n_false_incidents"] == 1
    assert out["false_alarms_per_day"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# alerting
# --------------------------------------------------------------------------

def test_group_incidents_merges_within_tolerance_and_splits_beyond():
    n = 100
    ts = _clock(n)
    flags = flags_at(n, [10, 11, 13, 60])  # gap of 2 merges, gap of 47 does not
    incidents = group_incidents(ts, flags, np.arange(n, dtype=float),
                                np.arange(n, dtype=float), max_gap_periods=3)
    assert len(incidents) == 2
    assert (incidents[0].start_idx, incidents[0].end_idx) == (10, 13)
    assert incidents[0].n_points == 3  # index 12 was never flagged
    assert incidents[1].start_idx == 60


def test_group_incidents_reports_peak_by_score():
    n = 50
    ts = _clock(n)
    flags = flags_at(n, [5, 6, 7])
    scores = np.zeros(n)
    scores[6] = 9.0
    incidents = group_incidents(ts, flags, scores, np.arange(n, dtype=float))
    assert incidents[0].peak_idx == 6
    assert incidents[0].peak_score == pytest.approx(9.0)


def test_group_incidents_empty_when_nothing_flagged():
    assert group_incidents(_clock(10), np.zeros(10, dtype=bool),
                           np.zeros(10), np.zeros(10)) == []


def test_threshold_for_rate_hits_the_requested_budget():
    scores = np.arange(1000, dtype=float)
    threshold = threshold_for_rate(scores, 0.01)
    assert (scores >= threshold).mean() == pytest.approx(0.01, abs=0.002)


def test_constant_scores_raise_no_alerts_by_threshold():
    """A flatline series scores identically everywhere and stands out nowhere.

    With an inclusive comparison the quantile lands on the tied block and
    flags the entire series, turning a 1% budget into 100%.
    """
    scores = np.full(1000, 3.0)
    flags = flags_from_scores(scores, threshold_for_rate(scores, 0.01))
    assert flags.sum() == 0


def test_budget_selection_honours_the_rate_despite_ties():
    scores = np.full(1000, 3.0)
    scores[500:510] = 9.0
    flags = flags_for_budget(scores, 0.01)
    assert flags.sum() == 10
    assert flags[500:510].all()  # the genuinely high scores win the budget


def test_budget_selection_is_exact_on_a_heavily_tied_surface():
    scores = np.repeat(np.arange(10, dtype=float), 100)  # 100-way ties
    flags = flags_for_budget(scores, 0.05)
    assert flags.sum() == 50


def test_budget_ties_break_towards_the_earliest_observation():
    scores = np.zeros(100)
    scores[[10, 50, 90]] = 5.0  # three-way tie, budget of two
    flags = flags_for_budget(scores, 0.02)
    assert list(np.flatnonzero(flags)) == [10, 50]


def test_budget_selection_ignores_non_finite_scores():
    scores = np.full(100, np.nan)
    scores[7] = 1.0
    flags = flags_for_budget(scores, 0.05)
    assert list(np.flatnonzero(flags)) == [7]


# --------------------------------------------------------------------------
# feature leakage guard
# --------------------------------------------------------------------------

def test_rolling_features_use_only_prior_observations():
    """A spike must not appear in its own rolling baseline."""
    n = 400
    frame = pd.DataFrame({"timestamp": _clock(n), "value": np.ones(n)})
    frame.loc[350, "value"] = 100.0
    featured = build_features(frame)

    spike = featured.index[featured["value"] == 100.0][0]
    # Baseline at the spike still reflects the flat history before it...
    assert featured.loc[spike, "rolling_mean_short"] == pytest.approx(1.0)
    assert featured.loc[spike, "residual_short"] == pytest.approx(99.0)
    # ...and the following row is the first to see it.
    assert featured.loc[spike + 1, "rolling_mean_short"] > 1.0


def test_build_features_drops_warmup_rows_rather_than_imputing():
    n = 500
    frame = pd.DataFrame({"timestamp": _clock(n),
                          "value": np.random.default_rng(0).normal(size=n)})
    featured = build_features(frame)
    assert len(featured) < n
    assert featured[["residual_long", "rolling_z"]].notna().all().all()
