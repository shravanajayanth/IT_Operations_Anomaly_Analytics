"""
Dashboard interaction tests (objective 7, synopsis section 4.8).

Streamlit renders client-side, so a page that returns HTTP 200 may still be
throwing on every interaction. `AppTest` executes the script in-process and
surfaces exceptions, which lets the budget slider, detector switch and
detail drawer actually be exercised.

Skipped when `results/scored/` is absent - run `scripts/run_experiments.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"

pytestmark = pytest.mark.skipif(
    not (ROOT / "results" / "scored").exists(),
    reason="no scored results; run scripts/run_experiments.py first",
)


def fresh() -> "AppTest":
    app = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def box(app, label):
    """Fetch a selectbox by label. Handles go stale after a rerun, so always
    re-fetch rather than holding a reference across `.run()`."""
    for widget in app.selectbox:
        if widget.label == label:
            return widget
    raise AssertionError(
        f"no selectbox {label!r}; found {[w.label for w in app.selectbox]}")


def test_app_renders():
    app = fresh()
    assert app.title[0].value == "IT Operations Anomaly Analytics"
    assert len(app.metric) == 6


@pytest.mark.parametrize("budget", [0.05, 0.5, 1.0, 2.5, 5.0])
def test_budget_slider_spans_its_whole_range(budget):
    app = fresh()
    app.slider[0].set_value(budget).run()
    assert not app.exception, [e.value for e in app.exception]


def test_raising_the_budget_raises_the_alert_count():
    """The core interaction: more budget, more alerts, more false alarms."""
    low = fresh()
    low.slider[0].set_value(0.5).run()
    high = fresh()
    high.slider[0].set_value(5.0).run()
    assert int(high.metric[0].value) > int(low.metric[0].value)
    assert float(high.metric[2].value) >= float(low.metric[2].value)


def test_every_detector_renders():
    app = fresh()
    for index in range(len(box(app, "Detector").options)):
        app = fresh()
        box(app, "Detector").select_index(index).run()
        assert not app.exception, [e.value for e in app.exception]


def test_unlabelled_control_series_degrades_to_na():
    """`art_flatline` carries no windows, so window metrics must not pretend."""
    app = fresh()
    series = box(app, "Series")
    if "art_flatline" not in series.options:
        pytest.skip("control corpus not downloaded")
    series.set_value("art_flatline").run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.metric[1].value == "n/a"  # windows caught
    assert app.metric[5].value == "n/a"  # NAB score


def test_merge_gap_cannot_increase_incident_count():
    app = fresh()
    app.slider[0].set_value(2.0).run()
    before = int(app.metric[0].value)
    app.slider[1].set_value(12).run()
    assert not app.exception, [e.value for e in app.exception]
    assert int(app.metric[0].value) <= before


def test_llm_narrative_falls_back_without_a_key():
    """The dashboard must stay usable - and deterministic - with no API key."""
    app = fresh()
    app.radio[0].set_value("LLM (optional)").run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.warning, "expected a fallback warning"
    assert any("What happened" in md.value for md in app.markdown)


def test_incident_detail_drawer_opens():
    app = fresh()
    inspect = box(app, "Inspect a candidate")
    assert inspect.options
    inspect.select_index(0).run()
    assert not app.exception, [e.value for e in app.exception]
