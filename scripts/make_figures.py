"""
Render the dissertation figures from `results/` into `results/figures/`.

    python scripts/make_figures.py

Figures
-------
    fig1_timeline.png          metric, labelled windows and incident candidates
    fig2_budget_tradeoff.png   detection vs alert cost across the budget ladder
    fig3_nab_vs_budget.png     NAB score against alert budget
    fig4_metric_distortion.png point-wise vs point-adjusted F1, and the ceiling

Colour: categorical slots 1-3 from a validated palette, assigned to detectors
in fixed order and never cycled. Series are both legended and direct-labelled,
so identity never rests on colour alone; detections additionally differ in
marker shape for the same reason. Text uses ink tokens, not series colour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from itops.alerting import flags_for_budget, group_incidents  # noqa: E402
from itops import evaluation as ev  # noqa: E402

# --- palette (validated: all-pairs CVD dE 9.2, normal-vision 24.0, light) ---
SERIES = {
    "A_rolling_zscore": "#2a78d6",      # slot 1 blue
    "B_iforest_raw": "#eb6834",         # slot 2 orange
    "C_iforest_contextual": "#1baf7a",  # slot 3 aqua
}
SHORT = {"A_rolling_zscore": "A  z-score",
         "B_iforest_raw": "B  IF raw",
         "C_iforest_contextual": "C  IF contextual"}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#9aa7b8"
GRID = "#e6e5e1"
GOOD = "#0ca30c"      # status: candidate inside a labelled window
CRITICAL = "#d03b3b"  # status: false alarm
WINDOW_FILL = "#f0efec"


def style(ax) -> None:
    """Recessive axes: no box, one soft gridline direction, ink-coloured text."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.xaxis.label.set_color(INK_SOFT)
    ax.yaxis.label.set_color(INK_SOFT)


def save(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(figure)
    print(f"  {path.relative_to(ROOT)}")


def fig_timeline(results: Path, series_name: str, detector: str, rate: float) -> None:
    path = results / "scored" / f"{series_name}.csv"
    if not path.exists():
        print(f"  skipped fig1: {path.name} not found")
        return
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    scores = frame[f"score_{detector}"].to_numpy(dtype=float)
    flags = flags_for_budget(scores, rate)
    incidents = group_incidents(frame["timestamp"], flags, scores, frame["value"])
    truth = frame["in_window"].to_numpy(dtype=bool)
    bounds = ev.segments(truth)

    in_window = np.zeros(len(frame), dtype=bool)
    for start, end in bounds:
        in_window[start:end + 1] = True

    figure, ax = plt.subplots(figsize=(11, 4.2))
    style(ax)
    for index, (start, end) in enumerate(bounds):
        ax.axvspan(frame["timestamp"].iloc[start], frame["timestamp"].iloc[end],
                   color=WINDOW_FILL, zorder=0,
                   label="Labelled anomaly window" if index == 0 else None)
    ax.plot(frame["timestamp"], frame["value"], color=SERIES[detector],
            linewidth=1.2, label="CPU utilization", zorder=2)
    if "rolling_mean_long" in frame:
        ax.plot(frame["timestamp"], frame["rolling_mean_long"], color=MUTED,
                linewidth=1.6, linestyle=(0, (2, 2)), label="Rolling baseline (24h)",
                zorder=3)

    hit = [i for i in incidents if in_window[i.start_idx:i.end_idx + 1].any()]
    miss = [i for i in incidents if not in_window[i.start_idx:i.end_idx + 1].any()]
    # Shape differs as well as colour, so the distinction survives CVD and print.
    if hit:
        ax.scatter([i.peak_time for i in hit], [i.peak_value for i in hit],
                   s=90, marker="o", facecolor=GOOD, edgecolor=SURFACE,
                   linewidth=2, zorder=5, label="Candidate inside window")
    if miss:
        ax.scatter([i.peak_time for i in miss], [i.peak_value for i in miss],
                   s=80, marker="X", facecolor=CRITICAL, edgecolor=SURFACE,
                   linewidth=1.5, zorder=5, label="Candidate outside window")

    # Count distinct windows detected - several candidates can land in one
    # window, so the number of hitting incidents is not the detection count.
    detected = sum(1 for start, end in bounds if flags[start:end + 1].any())

    ax.set_ylabel("CPU utilization")
    ax.set_title(
        f"{series_name}  -  {SHORT[detector]} at a {rate:.1%} alert budget\n"
        f"{detected} of {len(bounds)} windows caught, {len(miss)} false alarms",
        color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12),
              ncol=5, fontsize=8.5, labelcolor=INK_SOFT)
    save(figure, results / "figures" / "fig1_timeline.png")


def _ladder(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[summary["calibration"] == "budget"].sort_values("target_rate")


def _direct_label(ax, x, y, text, colour) -> None:
    """Relief for low-contrast slots: every line is also named at its end."""
    ax.annotate(text, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                color=colour, fontsize=8.5, va="center")


def fig_budget_tradeoff(results: Path, summary: pd.DataFrame) -> None:
    ladder = _ladder(summary)
    # Two panels rather than two y-axes: these measures share no scale, and a
    # dual-axis chart invites a comparison that the geometry does not support.
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    panels = [("window_recall_mean", "Window recall", axes[0]),
              ("false_alarms_per_day_mean", "False alarms per day", axes[1])]

    for column, title, ax in panels:
        style(ax)
        for detector, colour in SERIES.items():
            rows = ladder[ladder["detector"] == detector]
            ax.plot(rows["target_rate"] * 100, rows[column], color=colour,
                    linewidth=2, marker="o", markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label=SHORT[detector])
            _direct_label(ax, rows["target_rate"].iloc[-1] * 100,
                          rows[column].iloc[-1], SHORT[detector][0], colour)
        ax.set_xscale("log")
        ax.set_xlabel("Alert budget (% of observations flagged)")
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
        ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5])
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])

    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_SOFT)
    figure.suptitle(
        "Detection rises with alert budget - and so does the cost of reviewing it",
        color=INK, fontsize=11.5, x=0.09, ha="left", y=1.04)
    save(figure, results / "figures" / "fig2_budget_tradeoff.png")


def fig_nab_vs_budget(results: Path, summary: pd.DataFrame) -> None:
    ladder = _ladder(summary)
    figure, ax = plt.subplots(figsize=(8, 4.4))
    style(ax)
    ax.axhline(0, color=MUTED, linewidth=1.2, linestyle=(0, (3, 3)), zorder=1)
    # Anchored right, where the curves have long since dived below zero.
    ax.annotate("a detector that never fires scores 0", xy=(5.0, 0),
                xytext=(0, 8), textcoords="offset points", ha="right",
                color=INK_SOFT, fontsize=8.5)

    for detector, colour in SERIES.items():
        rows = ladder[ladder["detector"] == detector]
        ax.plot(rows["target_rate"] * 100, rows["nab_standard"], color=colour,
                linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=SHORT[detector], zorder=3)
        best = rows.loc[rows["nab_standard"].idxmax()]
        ax.scatter([best["target_rate"] * 100], [best["nab_standard"]], s=160,
                   facecolor="none", edgecolor=colour, linewidth=2, zorder=4)
        _direct_label(ax, rows["target_rate"].iloc[-1] * 100,
                      rows["nab_standard"].iloc[-1], SHORT[detector][0], colour)

    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2, 5])
    ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
    ax.set_xlabel("Alert budget (% of observations flagged)")
    ax.set_ylabel("NAB score (standard profile)")
    ax.set_title("NAB's cost model rewards restraint\n"
                 "Rings mark each detector's best budget; "
                 "past ~0.5% the false-alarm penalty dominates",
                 color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK_SOFT)
    save(figure, results / "figures" / "fig3_nab_vs_budget.png")


def fig_metric_distortion(results: Path, summary: pd.DataFrame, rate: float) -> None:
    rows = summary[(summary["calibration"] == "budget")
                   & (summary["target_rate"] == rate)]
    if rows.empty:
        print("  skipped fig4: no rows at the headline budget")
        return

    labels = [SHORT[d] for d in rows["detector"]]
    positions = np.arange(len(rows), dtype=float)
    width = 0.34
    figure, ax = plt.subplots(figsize=(8, 4.4))
    style(ax)

    # 2px surface gap between adjacent bars, via a small positional inset.
    ax.bar(positions - width / 2 - 0.012, rows["point_f1_mean"], width,
           color=MUTED, label="Point-wise F1 (against the window mask)")
    ax.bar(positions + width / 2 + 0.012, rows["pa_f1_mean"], width,
           color=SERIES["C_iforest_contextual"], label="Point-adjusted F1")

    ceiling = float(rows["point_recall_ceiling_mean"].iloc[0])
    ax.axhline(ceiling, color=CRITICAL, linewidth=1.6, linestyle=(0, (4, 3)), zorder=4)
    # Short tag only - the title already carries the full explanation, and a
    # long label here runs straight across the bars.
    ax.annotate(f"ceiling {ceiling:.2f}", xy=(-0.47, ceiling), xytext=(0, 6),
                textcoords="offset points", color=CRITICAL, fontsize=8.5,
                ha="left", zorder=6)

    # Value labels sit on a surface-coloured chip so the short bars stay
    # legible where they pass behind the ceiling rule.
    chip = dict(facecolor=SURFACE, edgecolor="none", pad=1.5)
    for offset, column in ((-width / 2 - 0.012, "point_f1_mean"),
                           (width / 2 + 0.012, "pa_f1_mean")):
        for position, value in zip(positions + offset, rows[column]):
            ax.annotate(f"{value:.3f}", xy=(position, value), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=8.5,
                        color=INK_SOFT, bbox=chip, zorder=6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"Point-wise F1 measures the wrong thing  (at a {rate:.1%} alert budget)\n"
        "NAB labels whole windows, so flagging 1% of points caps recall at "
        f"{ceiling:.2f} however good the detector is",
        color=INK, fontsize=11, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_SOFT)
    save(figure, results / "figures" / "fig4_metric_distortion.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--series", default="ec2_cpu_utilization_24ae8d")
    parser.add_argument("--detector", default="C_iforest_contextual")
    parser.add_argument("--rate", type=float, default=0.01)
    args = parser.parse_args()

    results = Path(args.results_dir)
    summary_path = results / "summary_by_detector.csv"
    if not summary_path.exists():
        print("no results found - run scripts/run_experiments.py first")
        return 1
    summary = pd.read_csv(summary_path)

    print("writing figures:")
    fig_timeline(results, args.series, args.detector, args.rate)
    fig_budget_tradeoff(results, summary)
    fig_nab_vs_budget(results, summary)
    fig_metric_distortion(results, summary, args.rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
