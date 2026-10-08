# IT Operations Anomaly Analytics

## Anomaly Detection and Performance Analytics in IT Operations Using Data Visualization

### MSc Data Science Project

This project focuses on detecting unusual patterns in IT operational performance data using statistical analysis, machine learning, feature engineering, and data visualization.

The project uses CPU utilization time-series data and applies different anomaly detection techniques to identify observations that differ from normal operational behavior.

## Objectives

- Analyze IT operational time-series data.
- Identify unusual patterns in CPU utilization.
- Implement Z-score based anomaly detection.
- Implement Isolation Forest for anomaly detection.
- Perform feature engineering on time-series data.
- Compare different anomaly detection methods.
- Evaluate models using Precision, Recall, and F1-score.
- Visualize detected anomalies.
- Develop an interactive Streamlit dashboard.

## Methodology

The project follows these steps:

1. Data collection
2. Data preprocessing
3. Exploratory data analysis
4. Z-score anomaly detection
5. Isolation Forest anomaly detection
6. Feature engineering
7. Feature-based Isolation Forest
8. Model evaluation
9. Model comparison
10. Interactive dashboard development

## Feature Engineering

The following features are created:

- Original value
- Rolling Mean
- Rolling Standard Deviation
- Percentage Change
- Lag-1 Value

## Machine Learning

The main machine learning algorithm used is **Isolation Forest**.

The project compares:

| Method | Description |
|---|---|
| Z-score Baseline | Statistical anomaly detection |
| Isolation Forest | Machine learning using original value |
| Feature-based Isolation Forest | Machine learning using engineered features |

The models are evaluated using:

- Precision
- Recall
- F1-score
- Confusion Matrix

## Dashboard

The project includes an interactive **Streamlit dashboard**.

The dashboard allows users to:

- Upload a CSV file
- Validate the uploaded data
- Perform data preprocessing
- Generate features
- Detect anomalies
- View performance graphs
- View detected anomaly records
- Download the analyzed dataset

### Required CSV Columns

The uploaded CSV should contain:

```text
timestamp
value
