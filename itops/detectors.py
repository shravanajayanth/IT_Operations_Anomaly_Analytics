"""
L3 - Detectors.

Every detector emits a *continuous* anomaly score (higher = more anomalous)
rather than a label. Decoupling scoring from thresholding is what makes the
threshold sweep, the PR curves and the dashboard's operating-point slider
possible; a detector that only returns labels is locked to whatever
`contamination` was guessed at fit time.

The three detectors form a deliberate ladder:

    A  RollingZScore   transparent statistical baseline
    B  IsolationForest on the raw value only - no temporal context
    C  IsolationForest on contextual features

B is expected to roughly tie A, because both can only isolate globally
extreme values. That tie is the control: any gap between B and C is
attributable to the feature representation, not to the model family.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .features import BASELINE_FEATURES, CONTEXT_FEATURES, RAW_FEATURES


class Detector:
    """Common interface: fit on a training slice, score any slice."""

    name: str = "detector"
    features: list[str] = []

    def fit(self, train: pd.DataFrame) -> "Detector":
        raise NotImplementedError

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Anomaly score per row; higher means more anomalous."""
        raise NotImplementedError

    def attribution(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-feature contribution estimate for the dashboard detail view."""
        return pd.DataFrame(index=frame.index)


@dataclass
class RollingZScore(Detector):
    """Baseline: absolute rolling z-score of the metric.

    Stateless - the rolling statistics are computed in the feature layer from
    strictly prior observations, so there is nothing to learn at fit time.
    Kept in the Detector interface so the harness treats all three uniformly.
    """

    name: str = "A_rolling_zscore"
    features: list[str] = field(default_factory=lambda: list(BASELINE_FEATURES))

    def fit(self, train: pd.DataFrame) -> "RollingZScore":
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        return np.abs(frame["rolling_z"].to_numpy(dtype=float))

    def attribution(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {"rolling_z": np.abs(frame["rolling_z"].to_numpy(dtype=float))},
            index=frame.index,
        )


@dataclass
class IsolationForestDetector(Detector):
    """Isolation Forest over a named feature set.

    `score_samples` returns higher values for *more normal* points, so it is
    negated here to keep the "higher = more anomalous" contract. `contamination`
    is left at its default because it only shifts the internal offset used by
    `predict`, which this class never calls.
    """

    name: str = "C_iforest_contextual"
    features: list[str] = field(default_factory=lambda: list(CONTEXT_FEATURES))
    n_estimators: int = 200
    max_samples: str | int = "auto"
    random_state: int = 42
    _model: IsolationForest | None = field(default=None, repr=False)
    _train_mean: pd.Series | None = field(default=None, repr=False)
    _train_std: pd.Series | None = field(default=None, repr=False)

    def fit(self, train: pd.DataFrame) -> "IsolationForestDetector":
        X = train[self.features]
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
        ).fit(X)
        # Retained for the attribution heuristic below.
        self._train_mean = X.mean()
        self._train_std = X.std(ddof=0).replace(0.0, np.nan)
        return self

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.name}: fit() must be called before score()")
        return -self._model.score_samples(frame[self.features])

    def attribution(self, frame: pd.DataFrame) -> pd.DataFrame:
        """How far each feature sits from its training distribution.

        A deliberately simple heuristic - absolute standardised deviation from
        the training mean - not a Shapley value. It answers "which signal looks
        unusual here?", which is what an analyst needs in the detail view, and
        it is cheap enough to recompute interactively.
        """
        if self._train_mean is None:
            return pd.DataFrame(index=frame.index)
        X = frame[self.features]
        return (X - self._train_mean).abs().div(self._train_std).fillna(0.0)


def default_detectors(random_state: int = 42) -> list[Detector]:
    """The A/B/C ladder described in the module docstring."""
    return [
        RollingZScore(),
        IsolationForestDetector(
            name="B_iforest_raw",
            features=list(RAW_FEATURES),
            random_state=random_state,
        ),
        IsolationForestDetector(
            name="C_iforest_contextual",
            features=list(CONTEXT_FEATURES),
            random_state=random_state,
        ),
    ]


DETECTOR_LABELS = {
    "A_rolling_zscore": "A - Rolling z-score (baseline)",
    "B_iforest_raw": "B - Isolation Forest (raw value)",
    "C_iforest_contextual": "C - Isolation Forest (contextual)",
}
