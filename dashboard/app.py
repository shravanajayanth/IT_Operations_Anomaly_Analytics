import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False



def get_openai_client():
    if not OPENAI_AVAILABLE:
        return None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def generate_ai_explanation(anomaly_df, all_df):
    client = get_openai_client()
    if client is None:
        return None, "OpenAI API key or package is not configured."
    anomaly_count = len(anomaly_df)
    total_count = len(all_df)
    anomaly_percentage = anomaly_count / total_count * 100 if total_count else 0
    recent = anomaly_df[["timestamp", "value", "rolling_mean", "rolling_std", "percentage_change", "lag_1"]].tail(10).copy()
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
        response = client.responses.create(model="gpt-6-luna", input=prompt, max_output_tokens=700)
        return response.output_text, None
    except Exception as e:
        return None, str(e)


def rule_based_explanation(anomaly_df, all_df):
    if anomaly_df.empty:
        return "### What Happened\nNo potential anomalies were detected in the selected period.\n\n### Risk Interpretation\nLow attention based on this dataset alone."
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IT Operations Anomaly Analytics",
    page_icon="📊",
    layout="wide"
)
