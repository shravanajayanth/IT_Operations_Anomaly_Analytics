import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ============================================================
# PAGE CONFIGURATION (must be the first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="IT Operations Anomaly Analytics",
    page_icon="📊",
    layout="wide",
)

DATA_PATH = "data/ec2_cpu_utilization_24ae8d.csv"
OPENAI_MODEL = "gpt-4o-mini"

ANOMALY_WINDOWS = [
    ("2014-02-26 13:45:00", "2014-02-27 06:25:00"),
    ("2014-02-27 08:55:00", "2014-02-28 01:35:00"),
]

FEATURES = ["value", "rolling_mean", "rolling_std", "percentage_change", "lag_1"]


# ============================================================
# OPENAI HELPERS
# ============================================================
def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None, "The openai package is not installed in this interpreter."
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, "No API key found. Add it to .streamlit/secrets.toml or set OPENAI_API_KEY."
    return OpenAI(api_key=api_key), None


def generate_ai_explanation(anomaly_df, all_df):
    client, err = get_openai_client()
    if client is None:
        return None, err

    anomaly_count = len(anomaly_df)
    total_count = len(all_df)
    anomaly_percentage = anomaly_count / total_count * 100 if total_count else 0
    recent = anomaly_df[
        ["timestamp", "value", "rolling_mean", "rolling_std", "percentage_change", "lag_1"]
    ].tail(10).copy()
    recent["timestamp"] = recent["timestamp"].astype(str)

    prompt = f"""You are an IT operations analytics assistant. Analyze this anomaly-detection summary. Do not claim an anomaly proves a system fault.

Total usable observations: {total_count}
Detected anomalies: {anomaly_count}
Anomaly percentage: {anomaly_percentage:.2f}%
Overall mean: {all_df['value'].mean():.4f}
Overall maximum: {all_df['value'].max():.4f}

Recent anomaly records:
{recent.to_string(index=False)}

Return exactly four sections:
1. What Happened
2. Possible Causes (2-4 possibilities)
3. Recommended Investigation (3-5 practical checks)
4. Risk Interpretation (Low/Medium/High attention, with reasoning)
Use cautious language such as may indicate/could be associated with.
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
        )
        return response.choices[0].message.content, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def rule_based_explanation(anomaly_df, all_df):
    if anomaly_df.empty:
        return (
            "### What Happened\nNo potential anomalies were detected in the selected period.\n\n"
            "### Risk Interpretation\nLow attention based on this dataset alone."
        )
    count = len(anomaly_df)
    pct = count / len(all_df) * 100 if len(all_df) else 0
    max_value = anomaly_df["value"].max()
    return f"""### What Happened
The Isolation Forest model identified **{count} potential anomalous observation(s)** ({pct:.2f}% of usable observations). The highest flagged value was **{max_value:.3f}**.

### Possible Causes
- Sudden increase in workload or system activity.
- Unusual application or batch process.
- Temporary resource contention.
- Data-quality or measurement issue.

### Recommended Investigation
1. Check system and application logs around anomaly timestamps.
2. Review CPU, memory, disk, network and process metrics.
3. Check deployments, scheduled jobs and configuration changes.
4. Compare with other monitoring dashboards.
5. Confirm whether the event was operational or a measurement issue.

### Risk Interpretation
**Low to Medium attention** based only on this time-series data. The detector is an analytical aid, not proof of an incident.
"""


# ============================================================
# DATA AND MODEL
# ============================================================
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["rolling_mean"] = df["value"].rolling(window=12).mean()
    df["rolling_std"] = df["value"].rolling(window=12).std()
    df["percentage_change"] = df["value"].pct_change()
    df["lag_1"] = df["value"].shift(1)

    df["ground_truth"] = False
    for start, end in ANOMALY_WINDOWS:
        mask = (df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))
        df.loc[mask, "ground_truth"] = True

    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    return df


@st.cache_data
def run_models(df, contamination, z_threshold):
    out = df.copy()

    # Baseline: z-score
    out["z_score"] = (out["value"] - out["value"].mean()) / out["value"].std()
    out["baseline_anomaly"] = out["z_score"].abs() > z_threshold

    # Isolation Forest with engineered features
    model = IsolationForest(contamination=contamination, random_state=42)
    out["prediction"] = model.fit_predict(out[FEATURES])
    out["feature_anomaly"] = out["prediction"] == -1
    return out


def evaluate(y_true, y_pred):
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=[False, True]),
    }


# ============================================================
# LOAD
# ============================================================
if not os.path.exists(DATA_PATH):
    st.error(
        f"Data file not found: {DATA_PATH}. "
        "Launch with `streamlit run app.py` from the project root folder."
    )
    st.stop()

raw_df = load_data(DATA_PATH)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Controls")
contamination = st.sidebar.slider("Isolation Forest contamination", 0.001, 0.05, 0.01, 0.001)
z_threshold = st.sidebar.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.5)

min_date = raw_df["timestamp"].min().date()
max_date = raw_df["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
show_baseline = st.sidebar.checkbox("Show z-score anomalies", value=False)
show_truth = st.sidebar.checkbox("Show ground-truth windows", value=True)

results = run_models(raw_df, contamination, z_threshold)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d = end_d = date_range[0] if isinstance(date_range, (tuple, list)) else date_range

view = results[
    (results["timestamp"].dt.date >= start_d) & (results["timestamp"].dt.date <= end_d)
].copy()
view_anoms = view[view["feature_anomaly"]]

# ============================================================
# HEADER AND KPIs
# ============================================================
st.title("📊 IT Operations Anomaly Analytics")
st.caption(
    "Prototype dashboard for exploring candidate anomalies. "
    "Flagged points are candidates for review, not confirmed incidents."
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Observations (selected)", f"{len(view):,}")
k2.metric("Isolation Forest anomalies", f"{len(view_anoms):,}")
k3.metric("Anomaly rate", f"{(len(view_anoms) / len(view) * 100) if len(view) else 0:.2f}%")
k4.metric("Peak CPU value", f"{view['value'].max():.2f}" if len(view) else "n/a")

# ============================================================
# TABS
# ============================================================
tab_overview, tab_eval, tab_detail, tab_ai = st.tabs(
    ["Overview", "Model Evaluation", "Anomaly Detail", "AI Explanation"]
)

# ---------- Overview ----------
with tab_overview:
    fig = go.Figure()

    if show_truth:
        for i, (s, e) in enumerate(ANOMALY_WINDOWS):
            fig.add_vrect(
                x0=s, x1=e, fillcolor="orange", opacity=0.15, line_width=0,
                annotation_text="NAB window" if i == 0 else None,
                annotation_position="top left",
            )

    fig.add_trace(go.Scatter(
        x=view["timestamp"], y=view["value"], mode="lines",
        name="CPU Utilization", line=dict(width=1),
    ))
    fig.add_trace(go.Scatter(
        x=view_anoms["timestamp"], y=view_anoms["value"], mode="markers",
        name="Isolation Forest anomalies", marker=dict(size=7, color="red"),
    ))
    if show_baseline:
        base = view[view["baseline_anomaly"]]
        fig.add_trace(go.Scatter(
            x=base["timestamp"], y=base["value"], mode="markers",
            name="Z-score anomalies", marker=dict(size=9, symbol="x", color="black"),
        ))

    fig.update_layout(
        title="CPU Utilization and Detected Anomalies",
        xaxis_title="Timestamp", yaxis_title="CPU Utilization",
        height=520, hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Value distribution")
        hist = go.Figure(go.Histogram(x=view["value"], nbinsx=50))
        hist.update_layout(height=300, xaxis_title="CPU Utilization", yaxis_title="Count")
        st.plotly_chart(hist, use_container_width=True)
    with c2:
        st.subheader("Method comparison (selected period)")
        both = int((view["baseline_anomaly"] & view["feature_anomaly"]).sum())
        st.table(pd.DataFrame({
            "Measure": ["Z-score anomalies", "Isolation Forest anomalies", "Detected by both"],
            "Count": [int(view["baseline_anomaly"].sum()), int(view["feature_anomaly"].sum()), both],
        }))

# ---------- Evaluation ----------
with tab_eval:
    st.subheader("Evaluation against NAB labelled windows (full dataset)")
    z_eval = evaluate(results["ground_truth"], results["baseline_anomaly"])
    f_eval = evaluate(results["ground_truth"], results["feature_anomaly"])

    comparison = pd.DataFrame({
        "Method": ["Z-score Baseline", "Feature-based Isolation Forest"],
        "Anomalies": [int(results["baseline_anomaly"].sum()), int(results["feature_anomaly"].sum())],
        "Precision": [z_eval["Precision"], f_eval["Precision"]],
        "Recall": [z_eval["Recall"], f_eval["Recall"]],
        "F1-score": [z_eval["F1-score"], f_eval["F1-score"]],
    })
    st.dataframe(comparison.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1-score": "{:.4f}"}),
                 use_container_width=True)

    cm1, cm2 = st.columns(2)
    labels = ["Normal", "Anomaly"]
    with cm1:
        st.write("Z-score confusion matrix (rows: actual, columns: predicted)")
        st.dataframe(pd.DataFrame(z_eval["cm"], index=labels, columns=labels))
    with cm2:
        st.write("Isolation Forest confusion matrix (rows: actual, columns: predicted)")
        st.dataframe(pd.DataFrame(f_eval["cm"], index=labels, columns=labels))

    st.caption(
        "NAB labels are wide windows, so point-wise precision and recall are conservative. "
        "Interpret results with the dataset limitations in mind."
    )

# ---------- Detail ----------
with tab_detail:
    st.subheader("Inspect a single flagged observation")
    if view_anoms.empty:
        st.info("No anomalies in the selected period.")
    else:
        options = view_anoms["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
        chosen = st.selectbox("Select anomaly timestamp", options)
        context = st.slider("Context window (observations either side)", 10, 300, 60, 10)

        idx = results.index[results["timestamp"] == pd.to_datetime(chosen)][0]
        lo, hi = max(0, idx - context), min(len(results), idx + context + 1)
        window = results.iloc[lo:hi]

        dfig = go.Figure()
        dfig.add_trace(go.Scatter(x=window["timestamp"], y=window["value"], mode="lines", name="CPU"))
        dfig.add_trace(go.Scatter(x=window["timestamp"], y=window["rolling_mean"],
                                  mode="lines", name="Rolling mean", line=dict(dash="dot")))
        pt = results.loc[[idx]]
        dfig.add_trace(go.Scatter(x=pt["timestamp"], y=pt["value"], mode="markers",
                                  name="Selected anomaly", marker=dict(size=12, color="red")))
        dfig.update_layout(height=420, xaxis_title="Timestamp", yaxis_title="CPU Utilization")
        st.plotly_chart(dfig, use_container_width=True)

        st.write("Feature values for the selected point")
        st.dataframe(pt[["timestamp"] + FEATURES + ["z_score"]], use_container_width=True)

    st.subheader("All flagged observations")
    st.dataframe(view_anoms[["timestamp"] + FEATURES], use_container_width=True, height=300)
    st.download_button(
        "Download anomalies as CSV",
        view_anoms[["timestamp"] + FEATURES].to_csv(index=False).encode("utf-8"),
        file_name="anomalies_selected_period.csv",
        mime="text/csv",
    )

# ---------- AI ----------
with tab_ai:
    st.subheader("Explanation of detected anomalies")
    st.caption("Generated text is advisory and may be wrong. Verify against logs and monitoring data.")

    client_check, client_err = get_openai_client()
    if client_check is None:
        st.warning(f"OpenAI is not available: {client_err}")

    if st.button("Generate AI explanation"):
        with st.spinner("Contacting OpenAI..."):
            text, error = generate_ai_explanation(view_anoms, view)
        if error:
            st.error(f"AI explanation failed: {error}")
            st.session_state["explanation"] = rule_based_explanation(view_anoms, view)
            st.session_state["explanation_source"] = "Rule-based fallback"
        else:
            st.session_state["explanation"] = text
            st.session_state["explanation_source"] = f"OpenAI ({OPENAI_MODEL})"

    if "explanation" in st.session_state:
        st.markdown(f"**Source:** {st.session_state['explanation_source']}")
        st.markdown(st.session_state["explanation"])
    else:
        st.info("Click the button to generate an explanation, or view the rule-based summary below.")
        with st.expander("Rule-based summary"):
            st.markdown(rule_based_explanation(view_anoms, view))
