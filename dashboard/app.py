# ============================================================
# IT OPERATIONS ANOMALY ANALYTICS
# Streamlit Dashboard
#
# Project:
# Anomaly Detection and Performance Analytics in IT Operations
# Using Data Visualization
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IT Operations Anomaly Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    .upload-box {
        padding: 30px;
        border-radius: 12px;
        border: 2px dashed #999999;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 📊 Project Dashboard")

    st.markdown("---")

    st.markdown(
        """
        **Project Title**

        Anomaly Detection and Performance Analytics in IT Operations Using Data Visualization
        """
    )

    st.markdown("---")

    st.markdown("### Dataset Requirements")

    st.write(
        "Your CSV file must contain:"
    )

    st.code(
        "timestamp\nvalue"
    )

    st.markdown("---")

    st.markdown("### Detection Method")

    st.write(
        "Isolation Forest"
    )

    st.markdown("---")

    st.markdown("### Feature Engineering")

    st.write(
        """
        • Rolling Mean

        • Rolling Standard Deviation

        • Percentage Change

        • Lag-1 Value
        """
    )

    st.markdown("---")

    st.caption(
        "MSc Data Science Project"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📊 IT Operations Anomaly Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Anomaly Detection and Performance Analytics in IT Operations Using Data Visualization
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📁 Upload IT Operations Dataset</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    Upload a CSV file containing IT operational time-series data.
    The file should contain <b>timestamp</b> and <b>value</b> columns.
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    help="Upload a CSV file containing timestamp and value columns."
)


# ============================================================
# BEFORE FILE UPLOAD
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="upload-box">

        <h3>📂 Upload a CSV file to begin</h3>

        <p>
        No dataset is loaded by default.
        Please upload your IT operations CSV file using the uploader above.
        </p>

        <p>
        Required columns:
        <b>timestamp</b> and <b>value</b>
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "The dashboard will remain empty until a CSV file is uploaded."
    )

    st.stop()


# ============================================================
# READ UPLOADED FILE
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"Unable to read the uploaded CSV file: {e}"
    )

    st.stop()


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "timestamp",
    "value"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    st.error(
        "The uploaded CSV is missing required column(s): "
        + ", ".join(missing_columns)
    )

    st.warning(
        "Please upload a CSV containing the following columns:"
    )

    st.code(
        "timestamp\nvalue"
    )

    st.stop()


# ============================================================
# DATA PREPROCESSING
# ============================================================

st.markdown(
    '<div class="section-title">🔧 Data Preparation</div>',
    unsafe_allow_html=True
)


# Convert timestamp

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)


# Convert value to numeric

df["value"] = pd.to_numeric(
    df["value"],
    errors="coerce"
)


# Remove invalid records

before_cleaning = len(df)

df = df.dropna(
    subset=[
        "timestamp",
        "value"
    ]
).copy()

after_cleaning = len(df)


# Sort data

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


if df.empty:

    st.error(
        "No valid records were found after data cleaning."
    )

    st.stop()


# ============================================================
# DATA CLEANING SUMMARY
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Uploaded Rows",
        before_cleaning
    )

with col2:

    st.metric(
        "Valid Rows",
        after_cleaning
    )

with col3:

    st.metric(
        "Removed Rows",
        before_cleaning - after_cleaning
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Feature Engineering</div>',
    unsafe_allow_html=True
)


# Rolling mean

df["rolling_mean"] = (
    df["value"]
    .rolling(window=12)
    .mean()
)


# Rolling standard deviation

df["rolling_std"] = (
    df["value"]
    .rolling(window=12)
    .std()
)


# Percentage change

df["percentage_change"] = (
    df["value"]
    .pct_change()
)


# Lag value

df["lag_1"] = (
    df["value"]
    .shift(1)
)


# Remove rows where engineered features
# are not available

df_model = (
    df.dropna(
        subset=[
            "rolling_mean",
            "rolling_std",
            "percentage_change",
            "lag_1"
        ]
    )
    .copy()
)


if len(df_model) < 20:

    st.error(
        "The uploaded dataset does not contain enough valid records "
        "for Isolation Forest analysis."
    )

    st.info(
        "Please upload a dataset containing at least 20 usable rows."
    )

    st.stop()


# ============================================================
# ISOLATION FOREST
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Anomaly Detection</div>',
    unsafe_allow_html=True
)


features = [
    "value",
    "rolling_mean",
    "rolling_std",
    "percentage_change",
    "lag_1"
]


X = df_model[
    features
]


model = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42
)


df_model["prediction"] = (
    model.fit_predict(X)
)


df_model["anomaly"] = (
    df_model["prediction"] == -1
)


# ============================================================
# ANOMALY SCORE
# ============================================================

df_model["anomaly_score"] = (
    -model.score_samples(X)
)


# ============================================================
# ANOMALY COUNT
# ============================================================

total_records = len(df_model)

anomaly_count = int(
    df_model["anomaly"].sum()
)

normal_count = (
    total_records - anomaly_count
)


anomaly_percentage = (
    anomaly_count / total_records * 100
)


# ============================================================
# DATE FILTER
# ============================================================

st.markdown(
    '<div class="section-title">📅 Date Range Filter</div>',
    unsafe_allow_html=True
)


min_date = (
    df_model["timestamp"].min().date()
)

max_date = (
    df_model["timestamp"].max().date()
)


date_range = st.date_input(
    "Select date range",
    value=[
        min_date,
        max_date
    ],
    min_value=min_date,
    max_value=max_date
)


if isinstance(date_range, tuple) or isinstance(date_range, list):

    if len(date_range) == 2:

        start_date = pd.to_datetime(
            date_range[0]
        )

        end_date = (
            pd.to_datetime(date_range[1])
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )

        filtered_df = df_model[
            (df_model["timestamp"] >= start_date)
            & (df_model["timestamp"] <= end_date)
        ].copy()

    else:

        filtered_df = df_model.copy()

else:

    filtered_df = df_model.copy()


# ============================================================
# FILTERED KPI VALUES
# ============================================================

filtered_total = len(
    filtered_df
)

filtered_anomalies = int(
    filtered_df["anomaly"].sum()
)

filtered_normal = (
    filtered_total - filtered_anomalies
)


if filtered_total > 0:

    filtered_anomaly_percentage = (
        filtered_anomalies
        / filtered_total
        * 100
    )

else:

    filtered_anomaly_percentage = 0


# ============================================================
# KPI DASHBOARD
# ============================================================

st.markdown(
    '<div class="section-title">📈 Performance Summary</div>',
    unsafe_allow_html=True
)


kpi1, kpi2, kpi3, kpi4 = st.columns(4)


with kpi1:

    st.metric(
        "Total Records",
        f"{filtered_total:,}"
    )


with kpi2:

    st.metric(
        "Normal Records",
        f"{filtered_normal:,}"
    )


with kpi3:

    st.metric(
        "Anomalies Detected",
        f"{filtered_anomalies:,}"
    )


with kpi4:

    st.metric(
        "Anomaly Rate",
        f"{filtered_anomaly_percentage:.2f}%"
    )


# ============================================================
# PERFORMANCE GRAPH
# ============================================================

st.markdown(
    '<div class="section-title">📊 Performance Monitoring</div>',
    unsafe_allow_html=True
)


fig = go.Figure()


# Normal/performance line

fig.add_trace(
    go.Scatter(
        x=filtered_df["timestamp"],
        y=filtered_df["value"],
        mode="lines",
        name="CPU Utilization",
        line=dict(
            width=2
        )
    )
)


# Anomaly points

anomaly_points = filtered_df[
    filtered_df["anomaly"]
]


if not anomaly_points.empty:

    fig.add_trace(
        go.Scatter(
            x=anomaly_points["timestamp"],
            y=anomaly_points["value"],
            mode="markers",
            name="Detected Anomalies",
            marker=dict(
                size=10,
                symbol="circle"
            )
        )
    )


fig.update_layout(

    title="IT Operations Performance and Detected Anomalies",

    xaxis_title="Timestamp",

    yaxis_title="Performance Value",

    hovermode="x unified",

    height=550,

    template="plotly_white"

)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ANOMALY SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Anomaly Summary</div>',
    unsafe_allow_html=True
)


if filtered_anomalies == 0:

    st.success(
        "No anomalies were detected in the selected date range."
    )

else:

    st.warning(
        f"{filtered_anomalies} anomalous record(s) "
        f"were detected in the selected period."
    )


# ============================================================
# ANOMALY RECORDS
# ============================================================

st.markdown(
    '<div class="section-title">🔍 Detected Anomaly Records</div>',
    unsafe_allow_html=True
)


if not anomaly_points.empty:

    anomaly_display = anomaly_points[
        [
            "timestamp",
            "value",
            "rolling_mean",
            "rolling_std",
            "percentage_change",
            "lag_1",
            "anomaly_score"
        ]
    ].copy()


    anomaly_display["percentage_change"] = (
        anomaly_display["percentage_change"] * 100
    )


    anomaly_display = (
        anomaly_display
        .rename(
            columns={
                "timestamp": "Timestamp",
                "value": "Value",
                "rolling_mean": "Rolling Mean",
                "rolling_std": "Rolling Std",
                "percentage_change": "Percentage Change (%)",
                "lag_1": "Previous Value",
                "anomaly_score": "Anomaly Score"
            }
        )
    )


    st.dataframe(
        anomaly_display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No anomaly records are available for the selected period."
    )


# ============================================================
# FEATURE INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">⚙️ Engineered Features</div>',
    unsafe_allow_html=True
)


feature_info = pd.DataFrame({

    "Feature": [

        "value",

        "rolling_mean",

        "rolling_std",

        "percentage_change",

        "lag_1"

    ],

    "Description": [

        "Original IT operational performance value",

        "Average value over the previous 12 observations",

        "Variation over the previous 12 observations",

        "Percentage change from the previous observation",

        "Previous observation value"

    ]

})


st.dataframe(
    feature_info,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PROCESSED DATA
# ============================================================

st.markdown(
    '<div class="section-title">📋 Processed Dataset</div>',
    unsafe_allow_html=True
)


with st.expander(
    "View processed dataset"
):

    st.dataframe(
        df_model,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">⬇️ Download Results</div>',
    unsafe_allow_html=True
)


download_df = df_model.copy()


download_df["anomaly"] = (
    download_df["anomaly"]
    .astype(bool)
)


csv_data = download_df.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="📥 Download Analyzed CSV",
    data=csv_data,
    file_name="it_operations_anomaly_results.csv",
    mime="text/csv"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">🤖 Model Information</div>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <div class="info-box">

    <b>Algorithm:</b> Isolation Forest<br><br>

    <b>Number of Estimators:</b> 100<br><br>

    <b>Contamination:</b> 1%<br><br>

    <b>Random State:</b> 42<br><br>

    <b>Features Used:</b> Value, Rolling Mean, Rolling Standard Deviation,
    Percentage Change and Lag-1 Value

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT INTERPRETATION
# ============================================================

st.markdown(
    '<div class="section-title">📝 Project Interpretation</div>',
    unsafe_allow_html=True
)


st.info(
    """
    The dashboard uses Isolation Forest to identify unusual patterns
    in IT operational performance data.

    Feature engineering is applied using rolling statistics,
    percentage change and lag values.

    Detected anomalies represent observations that differ
    significantly from the normal pattern in the dataset.

    An anomaly does not automatically indicate the exact root cause.
    In a real IT operations environment, detected events would
    require further investigation using system logs, network data,
    application metrics and other operational information.
    """
)


# ============================================================
# PROJECT CONCLUSION
# ============================================================

st.markdown(
    '<div class="section-title">🎯 Project Conclusion</div>',
    unsafe_allow_html=True
)


st.success(
    """
    The developed dashboard demonstrates how machine learning
    and data visualization can be used to identify unusual
    patterns in IT operational data.

    Isolation Forest analyzes multiple engineered features
    and highlights potentially anomalous observations.

    The interactive dashboard allows users to upload data,
    analyze performance, visualize anomalies and download
    the processed results.

    This approach can be extended in future work to include
    multiple IT metrics such as CPU utilization, memory usage,
    network traffic, disk utilization, application logs and
    real-time monitoring data.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "MSc Data Science | IT Operations Anomaly Analytics | "
    "Anomaly Detection and Performance Analytics in IT Operations"
)
