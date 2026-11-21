"""Streamlit app: pick a store and item, get a 28-day forecast with prediction intervals.

The app holds only a compact slice of recent history. Lag and rolling features are
computed from it by the same code used in training, and forecasts are produced
recursively, so nothing is typed in by hand.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demand.config import HISTORY_PATH, HORIZON, META_PATH, MODEL_PATH, REPORTS
from demand.data import load_panel
from demand.forecast import recursive_forecast
from demand.intervals import IntervalCalibrator

BLUE, GREY, INK = "#1F5FA8", "#9AA5B1", "#222222"

st.set_page_config(page_title="Demand Forecasting", page_icon="📈", layout="wide")


@st.cache_resource
def load_artifacts():
    panel = load_panel(HISTORY_PATH)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    calibrator = IntervalCalibrator.load(REPORTS / "interval_calibration.json")
    forecast = recursive_forecast(model, panel.values, panel.dates[-1], HORIZON, panel.stores, panel.items)
    return panel, meta, calibrator, forecast


@st.cache_data
def read_report(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


def band(fig, x, lower, upper, name, color=BLUE, opacity=0.18):
    fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=list(upper) + list(lower)[::-1],
                             fill="toself", fillcolor=color, opacity=opacity, line=dict(width=0),
                             name=name, hoverinfo="skip"))


panel, meta, calibrator, forecast = load_artifacts()
future_dates = pd.date_range(panel.dates[-1] + pd.Timedelta(days=1), periods=HORIZON)

st.title("📈 Retail Demand Forecasting")
st.caption(
    f"{meta['model']} with recursive multi-step forecasting and empirical prediction intervals. "
    f"History ends {panel.dates[-1].date()}; forecasts start the next day."
)

with st.sidebar:
    st.header("Forecast settings")
    store = st.selectbox("Store", sorted(set(panel.stores)))
    item = st.selectbox("Item", sorted(set(panel.items[panel.stores == store])))
    horizon = st.slider("Horizon (days)", 1, HORIZON, 14)
    level = st.select_slider("Interval", options=[0.5, 0.8, 0.9], value=0.8, format_func=lambda v: f"{v:.0%}")
    service = st.select_slider("Service level for stock", options=[0.75, 0.9, 0.95], value=0.95,
                               format_func=lambda v: f"{v:.0%}",
                               help="Chance you want to cover demand over the horizon without a stock-out.")

tab_forecast, tab_backtest, tab_insights = st.tabs(["Forecast", "How good is it?", "What drives demand"])

# ---------------------------------------------------------------- Forecast tab
with tab_forecast:
    col = panel.column(store, item)
    point = forecast[:horizon, col]
    lower, upper = calibrator.bounds(point, level)
    stock_per_day = calibrator.upper_quantile(point, service)

    recent = panel.values[-28:, col].sum()
    total = point.sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Expected units, next {horizon} days", f"{total:,.0f}",
              f"{(total / (recent * horizon / 28) - 1) * 100:+.1f}% vs last 28 days' pace")
    c2.metric(f"Daily range ({level:.0%})", f"{lower.mean():.0f} to {upper.mean():.0f}")
    c3.metric(f"Peak day: {future_dates[int(point.argmax())].strftime('%a %d %b')}", f"{point.max():.0f} units")
    # Per-day quantile: stocking this much each day covers that day's demand `service` of the time.
    # (Summing daily quantiles would badly overstate a multi-day total, so we don't.)
    c4.metric(f"Daily stock target ({service:.0%})", f"{stock_per_day.mean():.0f} / day",
              f"+{(stock_per_day - point).mean():.0f} buffer vs expected", delta_color="off")

    history_days = 90
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=panel.dates[-history_days:], y=panel.values[-history_days:, col],
                             name="History", line=dict(color=GREY, width=1.5)))
    band(fig, future_dates[:horizon], lower, upper, f"{level:.0%} interval")
    fig.add_trace(go.Scatter(x=future_dates[:horizon], y=point, name="Forecast",
                             line=dict(color=BLUE, width=2.5)))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Units sold",
                      legend=dict(orientation="h", y=1.08), hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    table = pd.DataFrame({"date": future_dates[:horizon].date, "forecast": point.round(1),
                          f"lower_{level:.0%}": lower.round(1), f"upper_{level:.0%}": upper.round(1)})
    with st.expander("Forecast table"):
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.download_button("Download CSV", table.to_csv(index=False), f"forecast_store{store}_item{item}.csv")

    with st.expander("Store total across all 50 items (point forecast only)"):
        in_store = panel.stores == store
        total_hist = panel.values[-60:, in_store].sum(axis=1)
        total_fc = forecast[:horizon][:, in_store].sum(axis=1)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=panel.dates[-60:], y=total_hist, name="History", line=dict(color=GREY)))
        fig2.add_trace(go.Scatter(x=future_dates[:horizon], y=total_fc, name="Forecast", line=dict(color=BLUE, width=2.5)))
        fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Units sold")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Intervals for totals need the correlation between items' errors, so none are shown here.")

# ---------------------------------------------------------------- Backtest tab
with tab_backtest:
    metrics = read_report("backtest_metrics.csv").set_index("model")
    best = meta["model"]
    naive = metrics.loc["Seasonal naive (last week)"]
    gain_mae = (1 - metrics.loc[best, "MAE"] / naive["MAE"]) * 100
    st.markdown(
        f"Scored by a **6-fold rolling-origin backtest**: each model is retrained on data before the "
        f"origin, then forecasts 28 days recursively with no peeking. **{best}** cuts MAE by "
        f"**{gain_mae:.0f}%** vs a seasonal-naive baseline."
    )
    st.dataframe(metrics[["MAE", "RMSE", "WAPE", "sMAPE", "Bias%"]].round(2), use_container_width=True)

    left, right = st.columns(2)
    left.image(str(REPORTS / "error_by_horizon.png"), caption="Error stays flat out to 28 days: it is dominated by day-to-day noise, not compounding")
    right.image(str(REPORTS / "forecast_example.png"), caption="Held-out forecast with 80% interval")

    coverage = read_report("interval_coverage.csv")
    st.subheader("Are the intervals honest?")
    st.caption("Calibrated on older folds, checked on the newest fold the calibration never saw.")
    st.dataframe(coverage.assign(nominal=coverage["nominal"].map("{:.0%}".format),
                                 empirical_coverage=coverage["empirical_coverage"].map("{:.1%}".format)),
                 hide_index=True)

    st.subheader("Held-out forecast for this series")
    fold = read_report("backtest_last_fold.csv")
    one = fold[(fold["store"] == store) & (fold["item"] == item)].copy()
    one["date"] = pd.to_datetime(one["date"])
    lo, hi = calibrator.bounds(one["forecast"].to_numpy(), level)
    fig3 = go.Figure()
    band(fig3, one["date"], lo, hi, f"{level:.0%} interval")
    fig3.add_trace(go.Scatter(x=one["date"], y=one["actual"], name="Actual", line=dict(color=INK), mode="lines+markers"))
    fig3.add_trace(go.Scatter(x=one["date"], y=one["forecast"], name="Forecast", line=dict(color=BLUE, width=2.5)))
    fig3.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Units sold",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig3, use_container_width=True)

    by_store = read_report("backtest_by_store.csv")
    st.subheader("Error by store")
    st.bar_chart(by_store.set_index("store")["WAPE"], y_label="WAPE (%)")

# ---------------------------------------------------------------- Insights tab
with tab_insights:
    st.markdown("SHAP values show how much each feature moves the forecast, in units sold.")
    if (REPORTS / "shap_summary.png").exists():
        a, b = st.columns(2)
        a.image(str(REPORTS / "shap_importance.png"))
        b.image(str(REPORTS / "shap_summary.png"))
    st.subheader("Limitations")
    st.markdown(
        "- Dataset has no promotions, prices, holidays or stock-outs, so none are modelled.\n"
        "- Sales are treated as demand; stock-outs would understate it.\n"
        "- Recursive forecasting can compound error; on this smooth data it barely does (see *How good is it?*), "
        "but spikier retail data would show it.\n"
        "- Intervals are marginal, calibrated on 6 backtest folds; a regime change would need recalibration."
    )
