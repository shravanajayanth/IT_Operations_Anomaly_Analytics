"""
Interactive prototype dashboard (objective 6).

    streamlit run dashboard/app.py

Four regions:

    1. KPI strip      detection and alert cost at the current operating point
    2. Timeline       metric, rolling baseline, labelled windows, incidents
    3. Control panel  series, detector and the alert-budget slider
    4. Detail drawer  one incident: local context and feature attribution

The slider is the point of the whole interface: dragging it moves the
operating point and updates detection and false-alarm cost together, so the
trade-off is visible rather than buried in a hyperparameter.

Scores are read from `results/scored/` so the dashboard never refits a model;
run `scripts/run_experiments.py` first.

This is an exploration aid. A flagged point is a candidate for investigation,
not a confirmed incident.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from itops.alerting import flags_for_budget, group_incidents, incidents_to_frame  # noqa: E402
from itops.detectors import DETECTOR_LABELS  # noqa: E402
from itops import evaluation as ev  # noqa: E402

RESULTS = ROOT / "results"
SCORED = RESULTS / "scored"

# Colour-blind safe, consistent in light and dark terminals.
INK = "#1f2a37"
METRIC = "#4C78A8"
BASELINE = "#9aa7b8"
WINDOW = "rgba(242, 142, 44, 0.18)"
ALERT_TRUE = "#2E7D5B"
ALERT_FALSE = "#C44E52"

st.set_page_config(page_title="IT Operations Anomaly Analytics",
                   page_icon="📊", layout="wide")


# --------------------------------------------------------------------------
# Optional narrative layer
# --------------------------------------------------------------------------

def rule_based_explanation(incidents: pd.DataFrame, frame: pd.DataFrame,
                           metrics: dict) -> str:
    """Deterministic summary. The default, so the dashboard is reproducible."""
    if incidents.empty:
        return ("### What happened\n"
                "No incident candidates were raised at this operating point. "
                "Lower the alert budget threshold to surface weaker signals.\n\n"
                "### Risk interpretation\n"
                "Low attention on the basis of this series alone.")

    true_hits = int(incidents["hits_window"].sum()) if "hits_window" in incidents else 0
    peak = incidents.loc[incidents["peak_score"].idxmax()]
    return f"""### What happened
The detector raised **{len(incidents)} incident candidate(s)** across
{metrics['span_days']:.1f} days ({metrics['alerts_per_day']:.2f} per day), of which
**{true_hits}** overlap a labelled anomaly window. The highest-scoring candidate
peaked at **{peak['peak_value']:.3f}** on **{peak['peak_time']}**, lasting
{peak['duration_minutes']:.0f} minutes.

### Possible causes
- A workload or traffic shift changing the normal operating level.
- A scheduled batch job or deployment coinciding with the window.
- Resource contention on a shared host.
- A measurement or collection artefact rather than a system event.

### Recommended investigation
1. Pull application and system logs spanning the candidate window.
2. Compare neighbouring metrics (CPU, memory, disk, network) for the same period.
3. Check deployment, configuration and scheduled-job history.
4. Confirm the collector was healthy - gaps and resets look like anomalies.
5. Record the outcome so the operating point can be tuned against real triage cost.

### Risk interpretation
**{'Medium' if metrics['false_alarms_per_day'] > 1 else 'Low to medium'} attention.**
At this budget the detector raises about
{metrics['false_alarms_per_day']:.2f} false alarms per day. These are candidates
for review, not confirmed incidents."""


def ai_explanation(incidents: pd.DataFrame, frame: pd.DataFrame,
                   metrics: dict) -> tuple[str | None, str | None]:
    """Optional LLM narration. Excluded from every reported metric."""
    try:
        from openai import OpenAI
    except ImportError:
        return None, "The `openai` package is not installed."
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        api_key = None
    if not api_key:
        return None, "No OPENAI_API_KEY configured in Streamlit secrets."

    recent = incidents.head(10)[
        ["start_time", "end_time", "peak_value", "peak_score", "n_points"]
    ].to_string(index=False)
    prompt = f"""You are an IT operations analytics assistant. Summarise this
anomaly-detection result. Never claim that a detection proves a system fault.

Observations scored: {metrics['n_scored']}
Incident candidates: {metrics['n_incidents']}
Alerts per day: {metrics['alerts_per_day']:.2f}
False alarms per day: {metrics['false_alarms_per_day']:.2f}
Window recall: {metrics['window_recall']}
Series mean: {frame['value'].mean():.4f}, max: {frame['value'].max():.4f}

Top incident candidates:
{recent}

Return exactly four sections: What Happened; Possible Causes (2-4);
Recommended Investigation (3-5 practical checks); Risk Interpretation
(Low/Medium/High attention, with reasoning). Use cautious language such as
"may indicate" or "could be associated with"."""
    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model="gpt-6-luna", input=prompt, max_output_tokens=700)
        return response.output_text, None
    except Exception as error:  # noqa: BLE001 - surfaced to the user as-is
        return None, str(error)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_data
def load_scored() -> dict[str, Path]:
    if not SCORED.exists():
        return {}
    return {path.stem: path for path in sorted(SCORED.glob("*.csv"))}


@st.cache_data
def load_frame(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def main() -> None:
    st.title("IT Operations Anomaly Analytics")
    st.caption(
        "Candidate anomalies in operational time-series data. "
        "Flagged points are candidates for investigation, not confirmed incidents."
    )

    scored = load_scored()
    if not scored:
        st.error(
            "No scored series found. Run:\n\n"
            "```\npython scripts/download_data.py\n"
            "python scripts/run_experiments.py\n```"
        )
        return

    # --- 3. Control panel -------------------------------------------------
    with st.sidebar:
        st.header("Controls")
        name = st.selectbox("Series", list(scored), index=0)
        frame = load_frame(str(scored[name]))

        # Options are the detector names themselves rather than the score
        # column names, so the widget's value matches what is displayed.
        detectors = [c[len("score_"):] for c in frame.columns
                     if c.startswith("score_")]
        detector_name = st.selectbox(
            "Detector", detectors,
            format_func=lambda d: DETECTOR_LABELS.get(d, d),
            index=len(detectors) - 1,
        )
        detector = f"score_{detector_name}"
        budget = st.slider(
            "Alert budget (% of observations flagged)",
            min_value=0.05, max_value=5.0, value=1.0, step=0.05,
            help="Moves the operating point. Watch recall and false alarms "
                 "per day move in opposite directions.",
        ) / 100.0
        gap = st.slider(
            "Incident merge gap (samples)", min_value=1, max_value=12, value=3,
            help="Flags closer together than this are treated as one incident.",
        )
        st.divider()
        narrative = st.radio("Narrative", ["Rule-based", "LLM (optional)"],
                             help="Rule-based is deterministic and is what the "
                                  "dissertation reports. The LLM layer is a "
                                  "presentation aid only.")

    scores = frame[detector].to_numpy(dtype=float)
    flags = flags_for_budget(scores, budget)
    incidents = group_incidents(frame["timestamp"], flags, scores,
                                frame["value"], max_gap_periods=gap)
    table = incidents_to_frame(incidents)

    truth = frame["in_window"].to_numpy(dtype=bool)
    bounds = ev.segments(truth)
    span_days = ((frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0])
                 .total_seconds() / 86400.0)
    window = ev.window_metrics(frame["timestamp"], flags, incidents, bounds, span_days)
    point = ev.point_metrics(truth, flags)
    nab = ev.nab_score_components(ev.alert_flags(len(frame), incidents), bounds)

    if not table.empty:
        in_window = np.zeros(len(frame), dtype=bool)
        for start, end in bounds:
            in_window[start:end + 1] = True
        table["hits_window"] = [
            bool(in_window[int(a):int(b) + 1].any())
            for a, b in zip(table["start_idx"], table["end_idx"])
        ]

    # --- 1. KPI strip -----------------------------------------------------
    columns = st.columns(6)
    columns[0].metric("Incident candidates", window["n_incidents"])
    columns[1].metric(
        "Windows caught",
        "n/a" if not bounds else f"{window['n_windows_detected']}/{window['n_windows']}")
    columns[2].metric("False alarms / day", f"{window['false_alarms_per_day']:.2f}")
    columns[3].metric(
        "MTTD",
        "n/a" if np.isnan(window["mttd_minutes"]) else f"{window['mttd_minutes']:.0f} min")
    columns[4].metric("Point-adjusted F1", f"{point['pa_f1']:.3f}")
    columns[5].metric(
        "NAB (standard)",
        "n/a" if not bounds
        else f"{ev.normalize(nab['raw'], nab['null'], nab['perfect']):.1f}")

    if bounds:
        st.caption(
            f"Raw point-wise F1 at this budget is {point['point_f1']:.3f}, against a "
            f"structural ceiling of {point['point_recall_ceiling']:.3f} on recall - "
            "NAB labels whole windows, so point-wise scoring understates detection. "
            "The point-adjusted and window figures above are the meaningful ones."
        )

    # --- 2. Timeline ------------------------------------------------------
    figure = go.Figure()
    for index, (start, end) in enumerate(bounds):
        figure.add_vrect(
            x0=frame["timestamp"].iloc[start], x1=frame["timestamp"].iloc[end],
            fillcolor=WINDOW, line_width=0, layer="below",
            annotation_text="labelled window" if index == 0 else None,
            annotation_position="top left",
        )
    figure.add_trace(go.Scatter(
        x=frame["timestamp"], y=frame["value"], name="Metric",
        line=dict(color=METRIC, width=1.2), hovertemplate="%{x}<br>%{y:.3f}<extra></extra>"))
    if "rolling_mean_long" in frame:
        figure.add_trace(go.Scatter(
            x=frame["timestamp"], y=frame["rolling_mean_long"], name="Rolling baseline",
            line=dict(color=BASELINE, width=1.0, dash="dot")))

    if not table.empty:
        for hits, colour, label in ((True, ALERT_TRUE, "Candidate in window"),
                                    (False, ALERT_FALSE, "Candidate outside window")):
            subset = table[table["hits_window"] == hits] if bounds else (
                table if not hits else table.iloc[0:0])
            if subset.empty:
                continue
            figure.add_trace(go.Scatter(
                x=subset["peak_time"], y=subset["peak_value"], mode="markers",
                name=label,
                marker=dict(color=colour, size=9, symbol="circle",
                            line=dict(color="white", width=1)),
                customdata=np.stack([subset["peak_score"], subset["n_points"]], axis=-1),
                hovertemplate=("%{x}<br>value %{y:.3f}"
                               "<br>score %{customdata[0]:.3f}"
                               "<br>%{customdata[1]} points<extra></extra>")))

    figure.update_layout(
        height=430, margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified", font=dict(color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis_title=None, yaxis_title="Metric value",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="rgba(0,0,0,0.06)")
    st.plotly_chart(figure, width="stretch")

    # --- 4. Detail drawer -------------------------------------------------
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Investigation queue")
        if table.empty:
            st.info("No incident candidates at this budget.")
        else:
            display = table[["start_time", "end_time", "duration_minutes",
                             "peak_value", "peak_score", "n_points"]].copy()
            if bounds:
                display.insert(0, "in window", table["hits_window"])
            st.dataframe(
                display.sort_values("peak_score", ascending=False),
                width="stretch", height=300, hide_index=True)

            choice = st.selectbox(
                "Inspect a candidate",
                options=list(range(len(table))),
                format_func=lambda i: (f"{table['start_time'].iloc[i]}  "
                                       f"(score {table['peak_score'].iloc[i]:.3f})"))
            incident = table.iloc[int(choice)]
            pad = max(30, int(incident["n_points"]) * 6)
            lo = max(0, int(incident["start_idx"]) - pad)
            hi = min(len(frame), int(incident["end_idx"]) + pad)
            local = frame.iloc[lo:hi]

            detail = go.Figure()
            detail.add_vrect(
                x0=incident["start_time"], x1=incident["end_time"],
                fillcolor="rgba(196, 78, 82, 0.12)", line_width=0, layer="below")
            detail.add_trace(go.Scatter(
                x=local["timestamp"], y=local["value"], name="Metric",
                line=dict(color=METRIC, width=1.4)))
            if "rolling_mean_short" in local:
                detail.add_trace(go.Scatter(
                    x=local["timestamp"], y=local["rolling_mean_short"],
                    name="Rolling mean", line=dict(color=BASELINE, width=1, dash="dot")))
            detail.update_layout(
                height=260, margin=dict(l=10, r=10, t=10, b=10),
                font=dict(color=INK), plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            st.plotly_chart(detail, width="stretch")

    with right:
        st.subheader("Interpretation")
        metrics = {
            "n_scored": len(frame), "span_days": span_days,
            "n_incidents": window["n_incidents"],
            "alerts_per_day": window["alerts_per_day"],
            "false_alarms_per_day": window["false_alarms_per_day"],
            "window_recall": ("n/a" if not bounds
                              else f"{window['window_recall']:.2f}"),
        }
        if narrative.startswith("LLM"):
            text, error = ai_explanation(table, frame, metrics)
            if text is None:
                st.warning(f"LLM narrative unavailable: {error} Falling back.")
                st.markdown(rule_based_explanation(table, frame, metrics))
            else:
                st.markdown(text)
                st.caption("Generated text. Not used in any reported metric.")
        else:
            st.markdown(rule_based_explanation(table, frame, metrics))


if __name__ == "__main__":
    main()
