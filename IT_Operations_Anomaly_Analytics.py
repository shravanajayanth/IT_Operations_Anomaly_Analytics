# ============================================================
# IT OPERATIONS ANOMALY ANALYTICS
# Anomaly Detection and Performance Analytics in IT Operations
# Using Data Visualization
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. CREATE RESULTS FOLDER
# ============================================================

os.makedirs("results", exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

file_path = "data/ec2_cpu_utilization_24ae8d.csv"

df = pd.read_csv(file_path)

print("\n========================================")
print("IT OPERATIONS ANOMALY ANALYTICS")
print("========================================")


# ============================================================
# 3. DATA PREPARATION
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp", "value"]
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


print("\nDATA TYPES:")
print(df.dtypes)

print("\nFIRST 5 ROWS:")
print(df.head())


# ============================================================
# 4. DATASET INFORMATION
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

df.info()

print("\nDATASET SHAPE:")
print(df.shape)

print("\nCOLUMN NAMES:")
print(df.columns.tolist())


# ============================================================
# 5. STATISTICAL SUMMARY
# ============================================================

print("\n========================================")
print("STATISTICAL SUMMARY")
print("========================================")

print(df["value"].describe())


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["timestamp"],
    df["value"],
    linewidth=1
)

plt.title(
    "CPU Utilization Over Time"
)

plt.xlabel(
    "Timestamp"
)

plt.ylabel(
    "CPU Utilization"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 7. Z-SCORE BASELINE ANOMALY DETECTION
# ============================================================

df["z_score"] = (
    (df["value"] - df["value"].mean())
    / df["value"].std()
)

df["baseline_anomaly"] = (
    df["z_score"].abs() > 3
)

zscore_count = int(
    df["baseline_anomaly"].sum()
)


print("\n========================================")
print("BASELINE ANOMALY DETECTION")
print("========================================")

print(
    "Number of anomalies detected:",
    zscore_count
)

print("\nANOMALY RECORDS:")

print(
    df[
        df["baseline_anomaly"]
    ][
        [
            "timestamp",
            "value",
            "z_score",
            "baseline_anomaly"
        ]
    ].head(10)
)


# ============================================================
# 8. ISOLATION FOREST USING ORIGINAL VALUE
# ============================================================

X = df[
    ["value"]
]

model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)

df["isolation_prediction"] = (
    model.fit_predict(X)
)

df["isolation_anomaly"] = (
    df["isolation_prediction"] == -1
)

isolation_count = int(
    df["isolation_anomaly"].sum()
)


print("\n========================================")
print("ISOLATION FOREST DETECTION")
print("========================================")

print(
    "Number of anomalies detected:",
    isolation_count
)

print("\nISOLATION FOREST ANOMALIES:")

print(
    df[
        df["isolation_anomaly"]
    ][
        [
            "timestamp",
            "value",
            "isolation_prediction",
            "isolation_anomaly"
        ]
    ].head(10)
)


# ============================================================
# 9. COMPARE Z-SCORE AND ISOLATION FOREST
# ============================================================

both_methods = (
    df["baseline_anomaly"]
    & df["isolation_anomaly"]
)

only_zscore = (
    df["baseline_anomaly"]
    & ~df["isolation_anomaly"]
)

only_isolation = (
    ~df["baseline_anomaly"]
    & df["isolation_anomaly"]
)

both_count = int(
    both_methods.sum()
)

zscore_only_count = int(
    only_zscore.sum()
)

isolation_only_count = int(
    only_isolation.sum()
)


print("\n========================================")
print("ANOMALY DETECTION COMPARISON")
print("========================================")

print(
    "Z-score anomalies:",
    zscore_count
)

print(
    "Isolation Forest anomalies:",
    isolation_count
)

print(
    "Detected by both methods:",
    both_count
)

print(
    "Detected only by Z-score:",
    zscore_only_count
)

print(
    "Detected only by Isolation Forest:",
    isolation_only_count
)


# ============================================================
# 10. CREATE GROUND TRUTH
# ============================================================

# Official NAB anomaly windows for this dataset

anomaly_windows = [

    (
        "2014-02-26 13:45:00",
        "2014-02-27 06:25:00"
    ),

    (
        "2014-02-27 08:55:00",
        "2014-02-28 01:35:00"
    )
]


df["ground_truth"] = False


for start, end in anomaly_windows:

    start = pd.to_datetime(start)

    end = pd.to_datetime(end)

    df.loc[
        (df["timestamp"] >= start)
        & (df["timestamp"] <= end),
        "ground_truth"
    ] = True


ground_truth_count = int(
    df["ground_truth"].sum()
)


print("\n========================================")
print("GROUND TRUTH")
print("========================================")

print(
    "Ground-truth anomaly observations:",
    ground_truth_count
)

print("\nGROUND-TRUTH RECORDS:")

print(
    df[
        df["ground_truth"]
    ].head(10)
)


# ============================================================
# 11. EVALUATE Z-SCORE BASELINE
# ============================================================

y_true = df["ground_truth"]

y_zscore = df["baseline_anomaly"]


z_precision = precision_score(
    y_true,
    y_zscore,
    zero_division=0
)

z_recall = recall_score(
    y_true,
    y_zscore,
    zero_division=0
)

z_f1 = f1_score(
    y_true,
    y_zscore,
    zero_division=0
)

z_cm = confusion_matrix(
    y_true,
    y_zscore
)


# ============================================================
# 12. EVALUATE ORIGINAL ISOLATION FOREST
# ============================================================

y_isolation = df["isolation_anomaly"]


i_precision = precision_score(
    y_true,
    y_isolation,
    zero_division=0
)

i_recall = recall_score(
    y_true,
    y_isolation,
    zero_division=0
)

i_f1 = f1_score(
    y_true,
    y_isolation,
    zero_division=0
)

i_cm = confusion_matrix(
    y_true,
    y_isolation
)


# ============================================================
# 13. DISPLAY INITIAL MODEL EVALUATION
# ============================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")


print("\nZ-SCORE BASELINE")

print(
    "Precision:",
    round(z_precision, 4)
)

print(
    "Recall:",
    round(z_recall, 4)
)

print(
    "F1-score:",
    round(z_f1, 4)
)

print("Confusion Matrix:")

print(z_cm)


print("\nISOLATION FOREST")

print(
    "Precision:",
    round(i_precision, 4)
)

print(
    "Recall:",
    round(i_recall, 4)
)

print(
    "F1-score:",
    round(i_f1, 4)
)

print("Confusion Matrix:")

print(i_cm)


# ============================================================
# 14. FEATURE ENGINEERING
# ============================================================

df["rolling_mean"] = (
    df["value"]
    .rolling(window=12)
    .mean()
)

df["rolling_std"] = (
    df["value"]
    .rolling(window=12)
    .std()
)

df["percentage_change"] = (
    df["value"]
    .pct_change()
)

df["lag_1"] = (
    df["value"]
    .shift(1)
)


# Remove rows where feature calculations
# are unavailable

df_features = (
    df.dropna()
    .copy()
)


print("\n========================================")
print("FEATURE ENGINEERING")
print("========================================")

print("\nFEATURE COLUMNS:")

print(
    df_features[
        [
            "timestamp",
            "value",
            "rolling_mean",
            "rolling_std",
            "percentage_change",
            "lag_1"
        ]
    ].head(15)
)


# ============================================================
# 15. ISOLATION FOREST WITH ENGINEERED FEATURES
# ============================================================

features = [

    "value",

    "rolling_mean",

    "rolling_std",

    "percentage_change",

    "lag_1"

]


X_features = (
    df_features[features]
)


feature_model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)


df_features["feature_prediction"] = (
    feature_model.fit_predict(
        X_features
    )
)


df_features["feature_anomaly"] = (
    df_features["feature_prediction"] == -1
)


feature_anomaly_count = int(
    df_features["feature_anomaly"].sum()
)


print("\n========================================")
print("ISOLATION FOREST WITH FEATURES")
print("========================================")

print(
    "Number of anomalies detected:",
    feature_anomaly_count
)


print("\nDETECTED ANOMALIES:")


print(
    df_features[
        df_features["feature_anomaly"]
    ][
        [
            "timestamp",
            "value",
            "rolling_mean",
            "rolling_std",
            "percentage_change",
            "lag_1"
        ]
    ].head(10)
)


# ============================================================
# 16. FEATURE-BASED MODEL EVALUATION
# ============================================================

# ground_truth is already present in df_features.
# Therefore, no merge operation is required.

evaluation_df = (
    df_features.copy()
)


y_true_features = (
    evaluation_df["ground_truth"]
)

y_feature_model = (
    evaluation_df["feature_anomaly"]
)


feature_precision = precision_score(
    y_true_features,
    y_feature_model,
    zero_division=0
)

feature_recall = recall_score(
    y_true_features,
    y_feature_model,
    zero_division=0
)

feature_f1 = f1_score(
    y_true_features,
    y_feature_model,
    zero_division=0
)

feature_cm = confusion_matrix(
    y_true_features,
    y_feature_model
)


print("\n========================================")
print("FEATURE-BASED MODEL EVALUATION")
print("========================================")

print(
    "Precision:",
    round(feature_precision, 4)
)

print(
    "Recall:",
    round(feature_recall, 4)
)

print(
    "F1-score:",
    round(feature_f1, 4)
)

print("Confusion Matrix:")

print(feature_cm)


# ============================================================
# 17. FINAL MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Method": [

        "Z-score Baseline",

        "Isolation Forest",

        "Feature-based Isolation Forest"

    ],

    "Anomalies Detected": [

        zscore_count,

        isolation_count,

        feature_anomaly_count

    ],

    "Precision": [

        z_precision,

        i_precision,

        feature_precision

    ],

    "Recall": [

        z_recall,

        i_recall,

        feature_recall

    ],

    "F1-score": [

        z_f1,

        i_f1,

        feature_f1

    ]

})


print("\n========================================")
print("FINAL MODEL COMPARISON")
print("========================================")

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 18. SAVE MODEL COMPARISON
# ============================================================

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)


# ============================================================
# 19. SAVE FEATURE ANOMALIES
# ============================================================

df_features[
    [
        "timestamp",
        "value",
        "rolling_mean",
        "rolling_std",
        "percentage_change",
        "lag_1",
        "feature_prediction",
        "feature_anomaly",
        "ground_truth"
    ]
].to_csv(
    "results/feature_anomalies.csv",
    index=False
)


print("\n========================================")
print("PROJECT RESULTS SAVED")
print("========================================")

print(
    "Saved: results/model_comparison.csv"
)

print(
    "Saved: results/feature_anomalies.csv"
)


# ============================================================
# 20. FINAL ANOMALY VISUALIZATION
# ============================================================

plt.figure(
    figsize=(14, 6)
)


# Plot CPU utilization

plt.plot(
    df_features["timestamp"],
    df_features["value"],
    label="CPU Utilization",
    linewidth=1
)


# Select detected anomalies

anomalies = (
    df_features[
        df_features["feature_anomaly"]
    ]
)


# Highlight anomalies

plt.scatter(
    anomalies["timestamp"],
    anomalies["value"],
    label="Detected Anomalies",
    s=35
)


plt.title(
    "IT Operations - CPU Utilization and Detected Anomalies"
)

plt.xlabel(
    "Timestamp"
)

plt.ylabel(
    "CPU Utilization"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


# Save visualization

plt.savefig(
    "results/anomaly_detection_plot.png",
    dpi=300
)


plt.show()


# ============================================================
# 21. PROJECT COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("PROJECT EXECUTION COMPLETED SUCCESSFULLY")
print("========================================")

print(
    "\nGenerated files:"
)

print(
    "1. results/model_comparison.csv"
)

print(
    "2. results/feature_anomalies.csv"
)

print(
    "3. results/anomaly_detection_plot.png"
)

print(
    "\nBest model based on F1-score:"
)

best_model = comparison.loc[
    comparison["F1-score"].idxmax(),
    "Method"
]

print(best_model)

print(
    "\nAnalysis completed successfully."
)
