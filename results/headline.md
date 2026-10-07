# Headline results

## Corpus

- series analysed: **29** (23 labelled, 6 unlabelled control)
- labelled anomaly windows: **49**
- observations: **157,416**
- mean labelled coverage: **9.8%** of each labelled file (NAB fixes this at 10%)

## Why point-wise F1 is the wrong metric

At a 1% alert budget against windows covering ~10% of each file, point-wise
recall is capped by construction. Measured over the labelled corpus at this
operating point:

| Detector | point F1 | point recall ceiling | point-adjusted F1 |
|---|---|---|---|
| A - Rolling z-score (baseline) | 0.025 | 0.088 | 0.811 |
| B - Isolation Forest (raw value) | 0.069 | 0.088 | 0.797 |
| C - Isolation Forest (contextual) | 0.036 | 0.088 | 0.829 |

## Detector comparison at a 1.0% alert budget

Equal alert volume for every detector, so the comparison is not
confounded by how much each one fires.

| Detector | window recall | point-adj. F1 | false alarms/day | MTTD (frac. of window) | NAB standard | NAB low-FP | NAB low-FN |
|---|---|---|---|---|---|---|---|
| A - Rolling z-score (baseline) | 0.785 ± 0.328 | 0.811 ± 0.288 | 1.67 | 0.31 | -39.4 | -153.0 | -0.3 |
| B - Isolation Forest (raw value) | 0.719 ± 0.312 | 0.797 ± 0.253 | 1.03 | 0.41 | -0.3 | -64.4 | 22.0 |
| C - Isolation Forest (contextual) | 0.800 ± 0.330 | 0.829 ± 0.293 | 1.45 | 0.35 | -18.6 | -112.3 | 14.3 |

MTTD is expressed as a fraction of window length. NAB windows open
before the labelled anomaly, so this is an upper bound on true latency.

## Threshold calibrated on training history only

The deployment-realistic number: the threshold is set on the
probationary period, so alert volume is whatever arrives.

| Detector | window recall | point-adj. F1 | actual alert rate | alerts/day | false alarms/day | NAB standard |
|---|---|---|---|---|---|---|
| A - Rolling z-score (baseline) | 0.694 | 0.686 | 1.52% | 2.63 | 2.29 | -53.7 |
| B - Isolation Forest (raw value) | 0.900 | 0.732 | 13.30% | 4.04 | 3.48 | -151.9 |
| C - Isolation Forest (contextual) | 0.891 | 0.838 | 4.53% | 4.98 | 4.20 | -178.4 |

## Unlabelled control corpus

`artificialNoAnomaly` contains no labelled anomalies, so every
alert raised there is a false alarm by definition.

| Detector | false alarms/day |
|---|---|
| A - Rolling z-score (baseline) | 2.19 |
| B - Isolation Forest (raw value) | 1.38 |
| C - Isolation Forest (contextual) | 2.11 |

## Alert budget trade-off (detector C, budget calibration)

| budget | window recall | point-adj. F1 | false alarms/day | NAB standard |
|---|---|---|---|---|
| 0.05% | 0.298 | 0.359 | 0.03 | 22.5 |
| 0.10% | 0.458 | 0.536 | 0.10 | 28.4 |
| 0.20% | 0.607 | 0.666 | 0.22 | 32.5 |
| 0.50% | 0.762 | 0.805 | 0.65 | 22.5 |
| 1.00% | 0.800 | 0.829 | 1.45 | -18.6 |
| 2.00% | 0.868 | 0.861 | 2.81 | -91.2 |
| 5.00% | 0.923 | 0.814 | 5.74 | -251.3 |

## Alert budget that NAB's cost model actually prefers

NAB charges every alert outside a window, so the score is highly
sensitive to alert volume. The budget maximising the standard-profile
score, per detector:

| Detector | best budget | NAB standard | window recall | point-adj. F1 | false alarms/day |
|---|---|---|---|---|---|
| A - Rolling z-score (baseline) | 0.10% | **28.0** | 0.425 | 0.477 | 0.12 |
| B - Isolation Forest (raw value) | 0.20% | **24.1** | 0.461 | 0.541 | 0.21 |
| C - Isolation Forest (contextual) | 0.20% | **32.5** | 0.607 | 0.666 | 0.22 |

At the 1% budget used as this study's headline - the rate the
original single-file script used as `contamination` - every detector
scores negative under NAB, because flagging 1% of a 120,000-row corpus
raises far more alerts than its cost model tolerates. That sensitivity
is itself a finding: NAB's score and point-adjusted F1 rank operating
points in opposite directions, so neither should be read alone.
