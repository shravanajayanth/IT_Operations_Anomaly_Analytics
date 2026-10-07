"""
Run the full protocol over every downloaded NAB series and write `results/`.

    python scripts/run_experiments.py

Outputs
-------
    results/dataset_summary.csv     obj. 1 - what was analysed
    results/per_series_metrics.csv  every (series, detector, calibration, budget)
    results/summary_by_detector.csv obj. 5 - headline comparison, mean +/- sd
    results/sweep.csv               threshold sweep behind the curves
    results/incidents.csv           obj. 6 - the triage queue
    results/scored/<name>.csv       per-point scores, consumed by the dashboard
    results/headline.md             the tables to paste into the dissertation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from itops.detectors import DETECTOR_LABELS, default_detectors  # noqa: E402
from itops.experiment import PRIMARY_RATE, aggregate, run_series  # noqa: E402
from itops.nab import available_keys, describe, load_labels, load_series  # noqa: E402


def write_headline(summary: pd.DataFrame, datasets: pd.DataFrame,
                   metrics: pd.DataFrame, path: Path, rate: float) -> None:
    """Render the dissertation-ready tables."""
    lines: list[str] = ["# Headline results", ""]

    labelled = datasets[datasets["n_windows"] > 0]
    lines += [
        "## Corpus", "",
        f"- series analysed: **{len(datasets)}** "
        f"({len(labelled)} labelled, {len(datasets) - len(labelled)} unlabelled control)",
        f"- labelled anomaly windows: **{int(datasets['n_windows'].sum())}**",
        f"- observations: **{int(datasets['rows'].sum()):,}**",
        f"- mean labelled coverage: **{labelled['labelled_fraction'].mean():.1%}** "
        "of each labelled file (NAB fixes this at 10%)",
        "",
        "## Why point-wise F1 is the wrong metric", "",
        "At a 1% alert budget against windows covering ~10% of each file, point-wise",
        "recall is capped by construction. Measured over the labelled corpus at this",
        "operating point:", "",
    ]

    primary = summary[(summary["calibration"] == "budget")
                      & (summary["target_rate"] == rate)]
    lines += ["| Detector | point F1 | point recall ceiling | point-adjusted F1 |",
              "|---|---|---|---|"]
    for _, row in primary.iterrows():
        lines.append(
            f"| {DETECTOR_LABELS.get(row['detector'], row['detector'])} "
            f"| {row['point_f1_mean']:.3f} "
            f"| {row['point_recall_ceiling_mean']:.3f} "
            f"| {row['pa_f1_mean']:.3f} |"
        )

    lines += ["", f"## Detector comparison at a {rate:.1%} alert budget", "",
              "Equal alert volume for every detector, so the comparison is not",
              "confounded by how much each one fires.", "",
              "| Detector | window recall | point-adj. F1 | false alarms/day | MTTD (frac. of window) | NAB standard | NAB low-FP | NAB low-FN |",
              "|---|---|---|---|---|---|---|---|"]
    for _, row in primary.iterrows():
        lines.append(
            f"| {DETECTOR_LABELS.get(row['detector'], row['detector'])} "
            f"| {row['window_recall_mean']:.3f} ± {row['window_recall_sd']:.3f} "
            f"| {row['pa_f1_mean']:.3f} ± {row['pa_f1_sd']:.3f} "
            f"| {row['false_alarms_per_day_mean']:.2f} "
            f"| {row['mttd_window_fraction_mean']:.2f} "
            f"| {row['nab_standard']:.1f} "
            f"| {row['nab_low_FP_rate']:.1f} "
            f"| {row['nab_low_FN_rate']:.1f} |"
        )
    lines += ["",
              "MTTD is expressed as a fraction of window length. NAB windows open",
              "before the labelled anomaly, so this is an upper bound on true latency.",
              ]

    train = summary[(summary["calibration"] == "train")
                    & (summary["target_rate"] == rate)]
    if not train.empty:
        lines += ["", "## Threshold calibrated on training history only", "",
                  "The deployment-realistic number: the threshold is set on the",
                  "probationary period, so alert volume is whatever arrives.", "",
                  "| Detector | window recall | point-adj. F1 | actual alert rate | alerts/day | false alarms/day | NAB standard |",
                  "|---|---|---|---|---|---|---|"]
        for _, row in train.iterrows():
            lines.append(
                f"| {DETECTOR_LABELS.get(row['detector'], row['detector'])} "
                f"| {row['window_recall_mean']:.3f} "
                f"| {row['pa_f1_mean']:.3f} "
                f"| {row['actual_rate_mean']:.2%} "
                f"| {row['alerts_per_day_mean']:.2f} "
                f"| {row['false_alarms_per_day_mean']:.2f} "
                f"| {row['nab_standard']:.1f} |"
            )

    if "control_false_alarms_per_day" in primary.columns:
        lines += ["", "## Unlabelled control corpus", "",
                  "`artificialNoAnomaly` contains no labelled anomalies, so every",
                  "alert raised there is a false alarm by definition.", "",
                  "| Detector | false alarms/day |", "|---|---|"]
        for _, row in primary.iterrows():
            value = row.get("control_false_alarms_per_day")
            if pd.notna(value):
                lines.append(
                    f"| {DETECTOR_LABELS.get(row['detector'], row['detector'])} "
                    f"| {value:.2f} |")

    lines += ["", "## Alert budget trade-off (detector C, budget calibration)", "",
              "| budget | window recall | point-adj. F1 | false alarms/day | NAB standard |",
              "|---|---|---|---|---|"]
    ladder = summary[(summary["calibration"] == "budget")
                     & (summary["detector"] == "C_iforest_contextual")]
    for _, row in ladder.sort_values("target_rate").iterrows():
        lines.append(
            f"| {row['target_rate']:.2%} "
            f"| {row['window_recall_mean']:.3f} "
            f"| {row['pa_f1_mean']:.3f} "
            f"| {row['false_alarms_per_day_mean']:.2f} "
            f"| {row['nab_standard']:.1f} |"
        )

    budget_rows = summary[summary["calibration"] == "budget"]
    lines += ["", "## Alert budget that NAB's cost model actually prefers", "",
              "NAB charges every alert outside a window, so the score is highly",
              "sensitive to alert volume. The budget maximising the standard-profile",
              "score, per detector:", "",
              "| Detector | best budget | NAB standard | window recall | point-adj. F1 | false alarms/day |",
              "|---|---|---|---|---|---|"]
    for detector, group in budget_rows.groupby("detector"):
        best = group.loc[group["nab_standard"].idxmax()]
        lines.append(
            f"| {DETECTOR_LABELS.get(detector, detector)} "
            f"| {best['target_rate']:.2%} "
            f"| **{best['nab_standard']:.1f}** "
            f"| {best['window_recall_mean']:.3f} "
            f"| {best['pa_f1_mean']:.3f} "
            f"| {best['false_alarms_per_day_mean']:.2f} |"
        )
    lines += ["",
              f"At the {rate:.0%} budget used as this study's headline - the rate the",
              "original single-file script used as `contamination` - every detector",
              "scores negative under NAB, because flagging 1% of a 120,000-row corpus",
              "raises far more alerts than its cost model tolerates. That sensitivity",
              "is itself a finding: NAB's score and point-adjusted F1 rank operating",
              "points in opposite directions, so neither should be read alone.", ""]

    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "nab"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--rate", type=float, default=PRIMARY_RATE,
                        help="headline alert budget, as a fraction of observations")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    scored_dir = results_dir / "scored"
    results_dir.mkdir(parents=True, exist_ok=True)
    scored_dir.mkdir(parents=True, exist_ok=True)

    labels_path = data_dir / "combined_windows.json"
    if not labels_path.exists():
        print("labels not found - run scripts/download_data.py first")
        return 1

    labels = load_labels(labels_path)
    keys = available_keys(data_dir, labels)
    if not keys:
        print(f"no series found under {data_dir}")
        return 1

    print(f"{len(keys)} series from {data_dir}\n")
    started = time.time()

    dataset_rows, metric_rows, sweep_rows, incident_frames = [], [], [], []
    skipped = []

    for position, key in enumerate(keys, start=1):
        series = load_series(data_dir, key, labels)
        dataset_rows.append(describe(series))

        metrics, sweep, scored, incidents = run_series(
            series, detectors=default_detectors(args.random_state),
            primary_rate=args.rate)
        if scored is None:
            skipped.append(key)
            print(f"[{position:2d}/{len(keys)}] {key}  SKIPPED (too short)")
            continue

        metric_rows.extend(metrics)
        sweep_rows.extend(sweep)
        incident_frames.extend(incidents)
        scored.to_csv(scored_dir / f"{series.name}.csv", index=False)

        headline = [m for m in metrics
                    if m["calibration"] == "budget" and m["target_rate"] == args.rate]
        parts = []
        for row in headline:
            tag = row["detector"].split("_")[0]
            recall = row["window_recall"]
            parts.append(f"{tag}:{'--' if pd.isna(recall) else f'{recall:.2f}'}")
        print(f"[{position:2d}/{len(keys)}] {key:<58} "
              f"windows={len(series.windows)} recall {' '.join(parts)}")

    datasets = pd.DataFrame(dataset_rows)
    metrics = pd.DataFrame(metric_rows)
    sweep = pd.DataFrame(sweep_rows)
    incidents = (pd.concat(incident_frames, ignore_index=True)
                 if incident_frames else pd.DataFrame())
    summary = aggregate(metrics)

    datasets.to_csv(results_dir / "dataset_summary.csv", index=False)
    metrics.to_csv(results_dir / "per_series_metrics.csv", index=False)
    summary.to_csv(results_dir / "summary_by_detector.csv", index=False)
    sweep.to_csv(results_dir / "sweep.csv", index=False)
    incidents.to_csv(results_dir / "incidents.csv", index=False)
    write_headline(summary, datasets, metrics,
                   results_dir / "headline.md", args.rate)

    print(f"\ncompleted in {time.time() - started:.1f}s"
          + (f", skipped {len(skipped)}" if skipped else ""))
    print(f"\n--- headline, budget calibration @ {args.rate:.1%} ---")
    view = summary[(summary["calibration"] == "budget")
                   & (summary["target_rate"] == args.rate)]
    columns = ["detector", "window_recall_mean", "pa_f1_mean", "point_f1_mean",
               "false_alarms_per_day_mean", "mttd_minutes_mean", "nab_standard"]
    print(view[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {results_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
