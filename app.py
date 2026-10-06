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
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 48px;
        font-weight: 700;
        color: #30313d;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #555;
        margin-bottom: 25px;
    }

    .upload-box {
        padding: 25px;
        border-radius: 12px;
        background-color: #f5f7fb;
        text-align: center;
        margin-top: 30px;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 25px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🎛️ Dashboard Controls")

st.sidebar.markdown("### 📁 Upload CSV Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=["csv"],
    help="CSV must contain timestamp and value columns."
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Required CSV columns**

    • timestamp  
    • value
    """
)


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📊 IT Operations Anomaly Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'An interactive dashboard for monitoring system performance '
    'and detecting unusual operational behaviour using machine '
    'learning and data visualization.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# BEFORE FILE UPLOAD
# ============================================================

if uploaded_file is None:

    st.markdown(
        """
        <div class="upload-box">

        <h2>📂 Upload Your Dataset</h2>

        <p>
        Upload an IT operations CSV file to begin
        anomaly detection and performance analysis.
        </p>

        <p>
        The analysis will appear here after you upload the file.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "👈 Please use **Upload CSV Dataset** in the sidebar to begin."
    )

    st.stop()


# ============================================================
# READ UPLOADED FILE
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as e:

    st.error(
        f"❌ Unable to read the uploaded CSV file: {e}"
    )

    st.stop()


# ============================================================
# FILE SUCCESS MESSAGE
# ============================================================

st.success(
    f"✅ Dataset uploaded successfully: **{uploaded_file.name}**"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
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
        "❌ The uploaded CSV is missing the following required "
        f"column(s): **{', '.join(missing_columns)}**"
    )

    st.info(
        "Your CSV should contain these columns:\n\n"
        "`timestamp` and `value`"
    )

    st.stop()


# ============================================================
# PREPROCESSING
# ============================================================

original_rows = len(df)

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

# Count invalid records
invalid_timestamp = df["timestamp"].isna().sum()
invalid_value = df["value"].isna().sum()

invalid_rows = max(
    invalid_timestamp,
    invalid_value
)

# Remove invalid rows
df = df.dropna(
    subset=["timestamp", "value"]
).copy()

# Sort
df = df.sort_values(
    "timestamp"
).reset_index(drop=True)


if df.empty:

    st.error(
        "❌ No valid records were found in the uploaded CSV."
    )

    st.stop()


# ============================================================
# DATA INFORMATION
# ============================================================

with st.expander("📄 Uploaded Dataset Information"):

    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        st.write("**File name**")
        st.write(uploaded_file.name)

    with info_col2:
        st.write("**Original records**")
        st.write(f"{original_rows:,}")

    with info_col3:
        st.write("**Valid records**")
        st.write(f"{len(df):,}")

    if invalid_rows > 0:

        st.warning(
            f"{invalid_rows} invalid record(s) were removed "
            "during preprocessing."
        )


# ============================================================
# FEATURE ENGINEERING
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


# Remove rows where feature values are unavailable
df_model = df.dropna().copy()


# ============================================================
# ISOLATION FOREST
# ============================================================

features = [
    "value",
    "rolling_mean",
    "rolling_std",
    "percentage_change",
    "lag_1"
]


if len(df_model) >= 20:

    model = IsolationForest(
        n_estimators=100,
        contamination=0.01,
        random_state=42
    )

    model.fit(
        df_model[features]
    )

    df_model["prediction"] = model.predict(
        df_model[features]
    )

    # -1 = anomaly
    #  1 = normal

    df_model["anomaly"] = (
        df_model["prediction"] == -1
    )

    model_status = "Isolation Forest successfully applied."

else:

    df_model["prediction"] = 1

    df_model["anomaly"] = False

    model_status = (
        "Not enough records for reliable Isolation Forest analysis. "
        "At least 20 usable records are recommended."
    )


# ============================================================
# CALCULATE RESULTS
# ============================================================

total_observations = len(df_model)

anomaly_count = int(
    df_model["anomaly"].sum()
)

average_value = float(
    df["value"].mean()
)

maximum_value = float(
    df["value"].max()
)

minimum_value = float(
    df["value"].min()
)


# ============================================================
# KPI SECTION
# ============================================================

st.markdown(
    '<div class="section-title">📌 Performance Summary</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Observations",
        f"{total_observations:,}"
    )

with col2:

    st.metric(
        "Detected Anomalies",
        f"{anomaly_count:,}"
    )

with col3:

    st.metric(
        "Average Value",
        f"{average_value:.3f}"
    )

with col4:

    st.metric(
        "Maximum Value",
        f"{maximum_value:.3f}"
    )


# ============================================================
# MODEL STATUS
# ============================================================

st.markdown("---")

if len(df_model) >= 20:

    st.success(
        f"🤖 {model_status}"
    )

else:

    st.warning(
        f"⚠️ {model_status}"
    )


# ============================================================
# DATE FILTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📅 Date Filter</div>',
    unsafe_allow_html=True
)

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

if min_date == max_date:

    selected_dates = (min_date, max_date)

    st.write(
        f"Date: **{min_date}**"
    )

else:

    selected_dates = st.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )


if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

    start_date = selected_dates[0]
    end_date = selected_dates[1]

    filtered_df = df_model[
        (
            df_model["timestamp"].dt.date
            >= start_date
        )
        &
        (
            df_model["timestamp"].dt.date
            <= end_date
        )
    ].copy()

else:

    filtered_df = df_model.copy()


# ============================================================
# PERFORMANCE GRAPH
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📈 Performance Analysis</div>',
    unsafe_allow_html=True
)


fig = go.Figure()


# Normal values
fig.add_trace(
    go.Scatter(
        x=filtered_df["timestamp"],
        y=filtered_df["value"],
        mode="lines",
        name="System Value"
    )
)


# Anomalies
anomaly_df = filtered_df[
    filtered_df["anomaly"]
].copy()


if not anomaly_df.empty:

    fig.add_trace(
        go.Scatter(
            x=anomaly_df["timestamp"],
            y=anomaly_df["value"],
            mode="markers",
            name="Detected Anomaly",
            marker=dict(
                size=11,
                symbol="x"
            )
        )
    )


fig.update_layout(
    title="System Performance Over Time",
    xaxis_title="Timestamp",
    yaxis_title="Value",
    height=500,
    hovermode="x unified"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# ANOMALY SUMMARY
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🚨 Anomaly Detection Summary</div>',
    unsafe_allow_html=True
)


if anomaly_count > 0:

    st.warning(
        f"⚠️ **{anomaly_count} potential anomalous observations "
        "were detected.**"
    )

    percentage = (
        anomaly_count / total_observations * 100
        if total_observations > 0
        else 0
    )

    st.write(
        f"Detected anomalies represent approximately "
        f"**{percentage:.2f}%** of the usable observations."
    )

else:

    st.success(
        "✅ No potential anomalies were detected."
    )


# ============================================================
# ANOMALY TABLE
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🔍 Detected Anomaly Records</div>',
    unsafe_allow_html=True
)


if not anomaly_df.empty:

    anomaly_display_columns = [
        "timestamp",
        "value",
        "rolling_mean",
        "rolling_std",
        "percentage_change",
        "lag_1"
    ]

    st.dataframe(
        anomaly_df[
            anomaly_display_columns
        ].reset_index(drop=True),
        use_container_width=True
    )

else:

    st.info(
        "No anomaly records are available for the selected period."
    )


# ============================================================
# ALL PROCESSED DATA
# ============================================================

st.markdown("---")

with st.expander("📋 View Processed Dataset"):

    processed_columns = [
        "timestamp",
        "value",
        "rolling_mean",
        "rolling_std",
        "percentage_change",
        "lag_1",
        "anomaly"
    ]

    st.dataframe(
        df_model[
            processed_columns
        ].reset_index(drop=True),
        use_container_width=True
    )


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">⬇️ Download Results</div>',
    unsafe_allow_html=True
)


download_columns = [
    "timestamp",
    "value",
    "rolling_mean",
    "rolling_std",
    "percentage_change",
    "lag_1",
    "anomaly"
]

download_df = df_model[
    download_columns
].copy()


csv_data = download_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="📥 Download Processed Results",
    data=csv_data,
    file_name="anomaly_detection_results.csv",
    mime="text/csv"
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🤖 Machine Learning Model</div>',
    unsafe_allow_html=True
)

st.write(
    "**Isolation Forest** is used for unsupervised anomaly detection."
)

st.write(
    "The model analyzes the following features:"
)

feature_col1, feature_col2 = st.columns(2)

with feature_col1:

    st.write("• Original value")
    st.write("• Rolling mean")
    st.write("• Rolling standard deviation")

with feature_col2:

    st.write("• Percentage change")
    st.write("• Previous observation (Lag 1)")


# ============================================================
# PROJECT CONCLUSION
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">📌 Project Conclusion</div>',
    unsafe_allow_html=True
)

st.write(
    "The uploaded IT operations dataset was processed through "
    "data cleaning, transformation and feature engineering."
)

st.write(
    "Isolation Forest was then applied to identify observations "
    "that differ from normal operational patterns."
)

st.write(
    "The detected anomalies are presented through interactive "
    "visualizations and tabular results, supporting further "
    "investigation of unusual IT operational behaviour."
)


# ============================================================
# SIDEBAR PROJECT INFORMATION
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown("### 📚 Project")

st.sidebar.write(
    "**Anomaly Detection and Performance Analytics "
    "in IT Operations Using Data Visualization**"
)

st.sidebar.write(
    "MSc Data Science Dissertation Project"
)

st.sidebar.write(
    "**Algorithm:** Isolation Forest"
)

st.sidebar.write(
    "**Feature Engineering:** Rolling Statistics, "
    "Percentage Change and Lag Values"
)