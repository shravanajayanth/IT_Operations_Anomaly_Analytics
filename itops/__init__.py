"""
IT Operations Anomaly Analytics.

A reproducible harness for comparing anomaly detectors on operational
time-series data, evaluated at window, point-adjusted and alert-cost levels.

Layers (see README):
    nab         L0  dataset loading and window labels
    features    L1  contextual feature construction
    detectors   L3  scoring models (baseline and Isolation Forest)
    alerting    L4  thresholding and incident grouping
    evaluation  L5  window / point-adjusted / NAB scoring
    experiment  L2  chronological protocol and orchestration
"""

__version__ = "0.2.0"
