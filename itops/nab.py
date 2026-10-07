"""
L0 - Loading and labelling of Numenta Anomaly Benchmark (NAB) series.

NAB publishes labels as *anomaly windows*, not anomalous points, with the
total window coverage of each file fixed at 10% of its length. Anything that
treats the expanded window mask as point-level ground truth is therefore
measuring the wrong quantity; see `itops.evaluation` for the metrics that
respect the window semantics.

NAB also defines a probationary period - the first 15% of a file, capped at
750 rows - which its own scorer excludes. We reuse that period as the
chronological training split, so no model ever sees scored data during fit.

Reference: Lavin, A. & Ahmad, S. (2015). Evaluating real-time anomaly
detection algorithms - the Numenta Anomaly Benchmark. IEEE ICMLA.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
LABELS_URL = f"{RAW_BASE}/labels/combined_windows.json"
DATA_URL = f"{RAW_BASE}/data"

PROBATION_FRACTION = 0.15
PROBATION_CAP = 750

#: Corpora used in this study. ``artificialNoAnomaly`` carries no labelled
#: windows and therefore acts as a pure false-positive control: every alert
#: raised on those files is, by definition, a false alarm.
CORPORA = ("realAWSCloudwatch", "realKnownCause", "artificialNoAnomaly")


@dataclass(frozen=True)
class Series:
    """One NAB file: its metric values and its labelled anomaly windows."""

    key: str  # e.g. "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv"
    frame: pd.DataFrame  # columns: timestamp, value
    windows: list[tuple[pd.Timestamp, pd.Timestamp]]

    @property
    def corpus(self) -> str:
        return self.key.split("/")[0]

    @property
    def name(self) -> str:
        return Path(self.key).stem

    @property
    def period(self) -> pd.Timedelta:
        """Median sampling interval, used to size rolling windows in rows."""
        return infer_period(self.frame["timestamp"])

    def window_mask(self) -> np.ndarray:
        return window_mask(self.frame["timestamp"], self.windows)

    def probation_end(self) -> int:
        return probation_end(len(self.frame))


def infer_period(timestamps: pd.Series) -> pd.Timedelta:
    """Median gap between consecutive observations.

    Median rather than mean so that a handful of collection gaps - which NAB
    files do contain - cannot inflate the inferred sampling rate.
    """
    deltas = pd.Series(timestamps).sort_values().diff().dropna()
    if deltas.empty:
        return pd.Timedelta(minutes=5)
    return deltas.median()


def probation_end(n_rows: int, fraction: float = PROBATION_FRACTION,
                  cap: int = PROBATION_CAP) -> int:
    """Index of the first scored row, following NAB's probationary rule."""
    return int(min(cap, np.ceil(fraction * n_rows)))


def as_index(timestamps) -> pd.DatetimeIndex:
    """Normalise any timestamp container to a positionally-indexed DatetimeIndex.

    Callers pass Series sliced out of reset frames, numpy arrays and plain
    lists; forcing one type here keeps positional indexing unambiguous.
    """
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(timestamps).to_numpy()))


def window_mask(timestamps, windows) -> np.ndarray:
    """Boolean mask of observations falling inside any labelled window.

    Used for point-adjusted scoring and for shading the dashboard timeline -
    *not* as point-level ground truth for raw precision/recall.
    """
    ts = as_index(timestamps)
    mask = np.zeros(len(ts), dtype=bool)
    for start, end in windows:
        mask |= np.asarray((ts >= start) & (ts <= end), dtype=bool)
    return mask


def window_bounds(timestamps, windows) -> list[tuple[int, int]]:
    """Translate time windows into inclusive ``(start_idx, end_idx)`` pairs.

    Windows that contain no observation at all are dropped, since a detector
    cannot be expected to flag a row that does not exist.
    """
    ts = as_index(timestamps)
    bounds: list[tuple[int, int]] = []
    for start, end in windows:
        inside = np.flatnonzero(np.asarray((ts >= start) & (ts <= end), dtype=bool))
        if inside.size:
            bounds.append((int(inside[0]), int(inside[-1])))
    return bounds


def load_labels(labels_path: str | Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Read ``combined_windows.json`` into parsed timestamp pairs."""
    with open(labels_path) as handle:
        raw = json.load(handle)
    return {
        key: [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in windows]
        for key, windows in raw.items()
    }


def load_series(data_dir: str | Path, key: str, labels: dict) -> Series:
    """Load a single NAB file and attach its windows."""
    frame = pd.read_csv(Path(data_dir) / key)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = (
        frame.sort_values("timestamp")
        .drop_duplicates(subset="timestamp", keep="first")
        .reset_index(drop=True)
    )
    return Series(key=key, frame=frame[["timestamp", "value"]],
                  windows=labels.get(key, []))


def available_keys(data_dir: str | Path, labels: dict,
                   corpora: tuple[str, ...] = CORPORA) -> list[str]:
    """Label keys whose CSV is present on disk, restricted to `corpora`."""
    data_dir = Path(data_dir)
    return sorted(
        key for key in labels
        if key.split("/")[0] in corpora and (data_dir / key).exists()
    )


def describe(series: Series) -> dict:
    """Row-level summary used for the dataset-documentation table (obj. 1)."""
    frame = series.frame
    mask = series.window_mask()
    period = series.period
    return {
        "key": series.key,
        "corpus": series.corpus,
        "name": series.name,
        "rows": len(frame),
        "start": frame["timestamp"].iloc[0],
        "end": frame["timestamp"].iloc[-1],
        "period_minutes": period.total_seconds() / 60.0,
        "span_days": (frame["timestamp"].iloc[-1]
                      - frame["timestamp"].iloc[0]).total_seconds() / 86400.0,
        "n_windows": len(series.windows),
        "labelled_points": int(mask.sum()),
        "labelled_fraction": float(mask.mean()),
        "probation_rows": series.probation_end(),
        "value_mean": float(frame["value"].mean()),
        "value_std": float(frame["value"].std()),
        "value_min": float(frame["value"].min()),
        "value_max": float(frame["value"].max()),
    }
