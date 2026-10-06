"""
Renewable Energy Forecasting & Asset Health Check — Streamlit dashboard.

- Wind Forecast tab: 1-hour-ahead forecasts (naive, XGBoost, LSTM) vs. actual output, Nov–Dec 2018
- Live Replay tab: replays recorded turbine readings as if arriving live, and runs the trained
  XGBoost model on demand (raw readings -> features -> forecast)
- Solar tab: inverter performance scorecard, estimated losses, and per-inverter daily detail

Run locally from the repository root:
    python -m streamlit run dashboard/app.py
"""

import json
import sys
from datetime import time as clock_time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor

# Make the feature pipeline next to this file importable, wherever the app is started from
sys.path.insert(0, str(Path(__file__).resolve().parent))
from wind_pipeline import HISTORY_STEPS, HORIZON_STEPS, build_features  # noqa: E402

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Data and models live in <repo root>/data/processed and <repo root>/models
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
MODEL_DIR = REPO_ROOT / "models"
REPO_URL = "https://github.com/trishhal26/renewable-energy-forecasting"

st.set_page_config(
    page_title="Renewable Energy Forecasting & Asset Health",
    page_icon="⚡",
    layout="wide",
)

# Colours (checked for colour-blind safety): one fixed colour per series
COLORS = {
    "actual": "#1f2937",         # near-black: the real output
    "naive": "#9aa0a6",          # gray, dotted: the baseline
    "xgboost_delta": "#1c7ed6",  # blue
    "lstm": "#0ca678",           # green
    "expected": "#6b7280",       # gray, dashed: what a typical inverter should produce
    "check": "#e8590c",          # orange: flagged inverters
    "ok": "#9aa0a6",             # gray: normal inverters
}

MODEL_NAMES = {
    "naive": "Naive (persistence)",
    "xgboost_delta": "XGBoost",
    "lstm": "LSTM",
}


# ---------------------------------------------------------------------------
# Data loading (cached so files are read once)
# ---------------------------------------------------------------------------

@st.cache_data
def load_wind():
    return pd.read_csv(
        DATA_DIR / "wind_test_predictions_all.csv",
        index_col="timestamp",
        parse_dates=True,
    )


@st.cache_data
def load_solar():
    scorecard = pd.read_csv(DATA_DIR / "solar_scorecard.csv")
    daily = pd.read_csv(DATA_DIR / "solar_daily.csv", parse_dates=["date"])
    return scorecard, daily


@st.cache_data
def load_readings():
    """Raw 10-minute turbine readings (late Oct–Dec 2018) used for the live replay."""
    return pd.read_csv(DATA_DIR / "wind_replay_readings.csv", index_col="timestamp", parse_dates=True)


@st.cache_resource
def load_forecaster():
    """The trained XGBoost model, its feature list, and the clipping limit used in training."""
    model = XGBRegressor()
    model.load_model(MODEL_DIR / "wind_xgboost_delta.json")
    features = json.loads((MODEL_DIR / "wind_xgboost_features.json").read_text())
    meta = json.loads((MODEL_DIR / "wind_xgboost_meta.json").read_text())
    return model, features, meta["max_power"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_table(df):
    """MAE, RMSE and skill vs. naive for each model on the given rows."""
    rows = []
    naive_rmse = np.sqrt(((df["actual"] - df["naive"]) ** 2).mean())
    for col, name in MODEL_NAMES.items():
        error = df["actual"] - df[col]
        rmse = np.sqrt((error ** 2).mean())
        rows.append({
            "Model": name,
            "MAE (kW)": round(error.abs().mean(), 1),
            "RMSE (kW)": round(rmse, 1),
            "Skill vs. naive": round(1 - rmse / naive_rmse, 3),
        })
    return pd.DataFrame(rows)


def format_inr(amount):
    """Format rupees in lakh / crore, as commonly written in India."""
    if amount >= 1e7:
        return f"₹{amount / 1e7:.2f} crore"
    if amount >= 1e5:
        return f"₹{amount / 1e5:.1f} lakh"
    return f"₹{amount:,.0f}"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("⚡ Renewable Energy Forecasting & Asset Health Check")
st.markdown(
    "Two questions an energy operator asks every day: **how much power will the wind turbine "
    "produce in the next hour**, and **are all the solar inverters producing what they should?** "
    f"Built on public Kaggle datasets · [Code and notebooks on GitHub]({REPO_URL})"
)

wind_tab, replay_tab, solar_tab = st.tabs(["💨 Wind Forecast", "🔴 Live Replay", "☀️ Solar Health Check"])


# ---------------------------------------------------------------------------
# Wind tab
# ---------------------------------------------------------------------------

with wind_tab:
    wind = load_wind()

    st.subheader("1-hour-ahead wind power forecast")
    st.caption(
        "Test period: Nov–Dec 2018 (never seen during training). All models predict the "
        "*change* in power over the next hour, using past observations only."
    )

    overall = score_table(wind)
    best = overall.loc[overall["Model"] == "XGBoost"].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Selected model", "XGBoost")
    c2.metric("Error reduction vs. naive (RMSE)", f"{best['Skill vs. naive']:.1%}")
    c3.metric("Test readings (10-min)", f"{len(wind):,}")

    # --- Controls ---
    min_date = wind.index.min().date()
    max_date = wind.index.max().date()

    col_a, col_b, col_c = st.columns([1, 1, 2])
    window_days = col_b.selectbox("Window length", [1, 3, 7, 14], index=2,
                                  format_func=lambda d: f"{d} day" + ("s" if d > 1 else ""))
    latest_start = max(min_date, max_date - pd.Timedelta(days=window_days - 1))
    start_date = col_a.date_input("Start date", value=min_date,
                                  min_value=min_date, max_value=latest_start)
    shown_models = col_c.multiselect(
        "Models to show", list(MODEL_NAMES), default=list(MODEL_NAMES),
        format_func=lambda c: MODEL_NAMES[c],
    )

    start = pd.Timestamp(start_date)
    window = wind[(wind.index >= start) & (wind.index < start + pd.Timedelta(days=window_days))]
    # Insert empty rows at missing timestamps so lines break at data gaps instead of bridging them
    window = window.asfreq("10min")

    # --- Chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=window.index, y=window["actual"], name="Actual",
        line=dict(color=COLORS["actual"], width=2),
    ))
    for col in shown_models:
        fig.add_trace(go.Scatter(
            x=window.index, y=window[col], name=MODEL_NAMES[col],
            line=dict(color=COLORS[col], width=1.5, dash="dot" if col == "naive" else "solid"),
        ))
    fig.update_layout(
        height=420,
        hovermode="x unified",
        yaxis_title="Power (kW)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    fig.update_traces(hovertemplate="%{y:,.0f} kW")
    st.plotly_chart(fig, width="stretch")

    # --- Scores ---
    left, right = st.columns(2)
    with left:
        st.markdown("**Whole test period**")
        st.dataframe(overall, hide_index=True, width="stretch")
    with right:
        st.markdown(f"**Selected window** ({window_days} days from {start_date})")
        st.dataframe(score_table(window.dropna()), hide_index=True, width="stretch")

    st.markdown(
        "**Takeaways**\n"
        "- Persistence (\"same as now\") is a strong baseline one hour ahead, because wind rarely changes drastically within an hour.\n"
        "- Predicting the *change* in power instead of its level more than doubled XGBoost's skill.\n"
        "- XGBoost and LSTM are essentially tied (~3% lower RMSE than persistence). XGBoost is selected: marginally better, faster, and easier to interpret.\n"
        "- The limit is the information available, not the model: larger gains would need weather-forecast inputs."
    )


# ---------------------------------------------------------------------------
# Live replay tab
# ---------------------------------------------------------------------------

with replay_tab:
    readings = load_readings()
    model, model_features, max_power = load_forecaster()

    st.subheader("Simulated real-time forecast")
    st.caption(
        "Replays the turbine's recorded readings as if they were arriving live. At the chosen "
        "moment, the app takes **only the readings up to that moment**, builds the features, and "
        "runs the trained XGBoost model to forecast power one hour ahead. Then you can reveal what "
        "actually happened. With a live data feed from a turbine, the same pipeline would run in real time."
    )

    step = pd.Timedelta(minutes=10)
    first_t = pd.Timestamp("2018-11-01 00:00")                     # start of the test period
    last_t = readings.index.max() - HORIZON_STEPS * step            # last moment with a known outcome

    if "replay_date" not in st.session_state:
        st.session_state.replay_date = pd.Timestamp("2018-11-16").date()
        st.session_state.replay_clock = clock_time(12, 0)

    def clamp(t):
        return min(max(t, first_t), last_t)

    def chosen_time():
        return pd.Timestamp.combine(st.session_state.replay_date, st.session_state.replay_clock)

    def move(minutes):
        """Step the replay clock forward or back (runs before the page redraws)."""
        t = clamp(chosen_time() + pd.Timedelta(minutes=minutes))
        st.session_state.replay_date = t.date()
        st.session_state.replay_clock = t.time()

    # --- Controls ---
    c1, c2, c3, c4, c5 = st.columns([1.3, 1, 0.8, 0.8, 0.8], vertical_alignment="bottom")
    c1.date_input("Date", key="replay_date", min_value=first_t.date(), max_value=last_t.date())
        # Dropdown of every 10-minute time of day (00:00, 00:10, ... 23:50): behaves the same in every Streamlit version
    times_of_day = [clock_time(h, m) for h in range(24) for m in range(0, 60, 10)]
    c2.selectbox("Time", times_of_day, key="replay_clock", format_func=lambda t: t.strftime("%H:%M"))
    c3.button("◀ 10 min", on_click=move, args=(-10,))
    c4.button("10 min ▶", on_click=move, args=(10,))
    c5.button("1 hour ⏩", on_click=move, args=(60,))
    reveal = st.toggle("Reveal what actually happened", value=True)

    now = clamp(chosen_time()).floor("10min")   # readings arrive on a 10-minute grid
    target_time = now + HORIZON_STEPS * step

    # Only the past: readings from (now − 24 h) up to and including now. Nothing after "now" is used.
    history = readings.loc[now - HISTORY_STEPS * step: now]
    features_now = build_features(history).loc[[now], model_features]

    if np.isnan(features_now.to_numpy()).any():
        st.warning(
            f"No forecast at {now:%d %b %Y, %H:%M}: the turbine has a data gap in the 24 hours "
            "before this moment, so the features can't be built. Step forward to a time with "
            "complete data."
        )
    else:
        # --- Run the model on demand ---
        power_now = float(features_now["power"].iloc[0])
        predicted_change = float(model.predict(features_now)[0])
        forecast = float(np.clip(power_now + predicted_change, 0, max_power))
        actual_later = readings["power"].get(target_time, np.nan)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Power now ({now:%H:%M})", f"{power_now:,.0f} kW")
        m2.metric(f"XGBoost forecast for {target_time:%H:%M}", f"{forecast:,.0f} kW",
                  delta=f"{forecast - power_now:+,.0f} kW vs. now", delta_color="off")
        m3.metric(f"Naive forecast for {target_time:%H:%M}", f"{power_now:,.0f} kW",
                  help="Persistence: assumes power stays the same as now.")
        if reveal and not np.isnan(actual_later):
            m4.metric(f"Actual at {target_time:%H:%M}", f"{actual_later:,.0f} kW")
        else:
            m4.metric(f"Actual at {target_time:%H:%M}", "hidden" if not reveal else "no reading")

        if reveal and not np.isnan(actual_later):
            xgb_miss = abs(forecast - actual_later)
            naive_miss = abs(power_now - actual_later)
            closer = "XGBoost" if xgb_miss < naive_miss else "Naive" if naive_miss < xgb_miss else "Neither"
            st.markdown(
                f"XGBoost missed by **{xgb_miss:,.0f} kW**; naive missed by **{naive_miss:,.0f} kW** "
                f"→ closer this time: **{closer}**. Single forecasts vary; the Wind Forecast tab shows "
                "performance over the whole test period."
            )

        # --- Chart: last 6 hours, the forecast, and (optionally) what happened next ---
        past = readings.loc[now - pd.Timedelta(hours=6): now, "power"]
        chart = go.Figure()
        chart.add_trace(go.Scatter(
            x=past.index, y=past, name="Readings so far",
            line=dict(color=COLORS["actual"], width=2), hovertemplate="%{y:,.0f} kW",
        ))
        chart.add_trace(go.Scatter(
            x=[now, target_time], y=[power_now, power_now], name="Naive forecast",
            mode="lines", line=dict(color=COLORS["naive"], width=2, dash="dot"),
            hovertemplate="%{y:,.0f} kW",
        ))
        chart.add_trace(go.Scatter(
            x=[now, target_time], y=[power_now, forecast], name="XGBoost forecast",
            mode="lines+markers", line=dict(color=COLORS["xgboost_delta"], width=2, dash="dash"),
            marker=dict(size=[0, 12], color=COLORS["xgboost_delta"]),
            hovertemplate="%{y:,.0f} kW",
        ))
        if reveal:
            future = readings.loc[now: target_time, "power"]
            chart.add_trace(go.Scatter(
                x=future.index, y=future, name="What actually happened",
                line=dict(color=COLORS["actual"], width=2), opacity=0.35,
                hovertemplate="%{y:,.0f} kW",
            ))
        chart.add_shape(type="line", x0=now, x1=now, y0=0, y1=1, yref="paper",
                        line=dict(color="#444444", width=1, dash="dot"))
        chart.add_annotation(x=now, y=1, yref="paper", text="now", showarrow=False,
                             yanchor="bottom", font=dict(size=12))
        chart.update_layout(
            height=420,
            hovermode="x unified",
            yaxis_title="Power (kW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.06, x=0),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(chart, width="stretch")

    st.caption(
        "Only XGBoost runs live here. The LSTM needs PyTorch, which is too large for free hosting; "
        "its results are in the Wind Forecast tab. The feature pipeline used here was verified to "
        "reproduce the training features exactly (notebook 07)."
    )


# ---------------------------------------------------------------------------
# Solar tab
# ---------------------------------------------------------------------------

with solar_tab:
    scorecard, daily = load_solar()

    st.subheader("Which inverters are underperforming, and what does it cost?")
    st.caption(
        "A model learns what a typical inverter produces for the weather and time of day. Each "
        "inverter's actual output is compared with that expectation. Performance ratio = actual ÷ "
        "expected energy; below 0.95 is flagged **Check**."
    )

    # --- Headline numbers (both plants) ---
    tariff = st.number_input("Tariff assumption (₹ per kWh)", min_value=1.0, max_value=10.0,
                             value=3.0, step=0.5)
    n_days = daily["date"].nunique()
    flagged = scorecard[scorecard["status"] == "Check"]
    lost_kwh = flagged["lost_kwh"].sum()
    lost_rs = lost_kwh * tariff

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Inverters flagged", f"{len(flagged)} of {len(scorecard)}")
    k2.metric(f"Energy lost ({n_days} days)", f"{lost_kwh:,.0f} kWh")
    k3.metric(f"Estimated loss ({n_days} days)", format_inr(lost_rs))
    k4.metric("Annualised (indicative)", format_inr(lost_rs * 365 / n_days))

    # --- Per-plant view ---
    plant = st.radio("Plant", sorted(scorecard["plant"].unique()), horizontal=True)
    plant_sc = scorecard[scorecard["plant"] == plant].sort_values("performance_ratio", ascending=False)

    p1, p2, p3 = st.columns(3)
    p1.metric("Flagged in this plant",
              f"{(plant_sc['status'] == 'Check').sum()} of {len(plant_sc)}")
    p2.metric("Median performance ratio", f"{plant_sc['performance_ratio'].median():.3f}")
    p3.metric("Total offline hours in sunlight", f"{plant_sc['downtime_h'].sum():,.0f} h")

    # Dot plot: one dot per inverter (worst at the top)
    order = plant_sc["inverter"].tolist()
    dot = go.Figure()
    for status, color in [("Check", COLORS["check"]), ("OK", COLORS["ok"])]:
        part = plant_sc[plant_sc["status"] == status]
        dot.add_trace(go.Scatter(
            x=part["performance_ratio"], y=part["inverter"], mode="markers",
            name="Check (< 0.95)" if status == "Check" else "OK",
            marker=dict(color=color, size=11, line=dict(color="white", width=1)),
            customdata=np.stack([part["lost_kwh"], part["downtime_h"]], axis=-1) if len(part) else None,
            hovertemplate=(
                "<b>%{y}</b><br>Performance ratio: %{x:.3f}"
                "<br>Lost energy: %{customdata[0]:,.0f} kWh"
                "<br>Offline in sunlight: %{customdata[1]:.1f} h<extra></extra>"
            ),
        ))
    dot.add_vline(x=1.0, line_color="#444444", line_width=1)
    dot.add_vline(x=0.95, line_color="#444444", line_width=1, line_dash="dash",
                  annotation_text="0.95 threshold", annotation_position="top left")
    dot.update_layout(
        height=max(420, 24 * len(plant_sc)),
        xaxis_title="Performance ratio (actual ÷ expected)",
        yaxis=dict(categoryorder="array", categoryarray=order, tickfont=dict(size=11)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(dot, width="stretch")

    # --- Inverter detail ---
    st.markdown("**Inverter detail**")
    worst_first = plant_sc.sort_values("performance_ratio")
    inverter = st.selectbox(
        "Choose an inverter (worst first)", worst_first["inverter"].tolist(),
        format_func=lambda inv: (
            f"{inv} — ratio "
            f"{worst_first.loc[worst_first['inverter'] == inv, 'performance_ratio'].iloc[0]:.3f}"
        ),
    )
    inv_row = plant_sc[plant_sc["inverter"] == inverter].iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Status", inv_row["status"])
    d2.metric("Performance ratio", f"{inv_row['performance_ratio']:.3f}")
    d3.metric("Lost energy", f"{inv_row['lost_kwh']:,.0f} kWh")
    d4.metric("Offline in sunlight", f"{inv_row['downtime_h']:.1f} h")

    inv_daily = daily[(daily["plant"] == plant) & (daily["inverter"] == inverter)]
    detail = go.Figure()
    detail.add_trace(go.Scatter(
        x=inv_daily["date"], y=inv_daily["expected_kwh"], name="Expected (typical inverter)",
        mode="lines+markers", line=dict(color=COLORS["expected"], width=2, dash="dash"),
        marker=dict(size=8),
    ))
    detail.add_trace(go.Scatter(
        x=inv_daily["date"], y=inv_daily["actual_kwh"], name="Actual",
        mode="lines+markers", line=dict(color=COLORS["xgboost_delta"], width=2),
        marker=dict(size=8),
    ))
    detail.update_layout(
        height=360,
        hovermode="x unified",
        yaxis_title="Daily energy (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    detail.update_traces(hovertemplate="%{y:,.0f} kWh")
    st.plotly_chart(detail, width="stretch")

    # --- Full scorecard ---
    with st.expander("Full scorecard for this plant"):
        table = worst_first[["inverter", "status", "performance_ratio", "peer_ratio",
                             "actual_kwh", "expected_kwh", "lost_kwh", "downtime_h"]].rename(columns={
            "inverter": "Inverter", "status": "Status", "performance_ratio": "Performance ratio",
            "peer_ratio": "Peer ratio (model-free check)", "actual_kwh": "Actual (kWh)",
            "expected_kwh": "Expected (kWh)", "lost_kwh": "Lost (kWh)", "downtime_h": "Offline (h)",
        })
        st.dataframe(table.round(3), hide_index=True, width="stretch")

    with st.expander("How to read this, and limitations"):
        st.markdown(
            "- **Plant 1 is healthy:** only 2 inverters are flagged.\n"
            "- **Plant 2 has a plant-wide problem:** most inverters show repeated zero-output periods in good "
            "sunlight, about 60× more often than Plant 1, which points to a shared plant-level cause.\n"
            "- **Validated two ways:** the model-based ranking matches a model-free comparison with each "
            "plant's median inverter (Spearman ρ = 0.971).\n"
            "- A real outage can't be told apart from a logger recording 0, so these are *zero-output periods*.\n"
            "- Costs depend on the tariff assumption; the annualised figure assumes the same weather and "
            "faults all year and is indicative only. Plant 2's yardstick is less precise (R² 0.79 vs. the "
            "peer median), so its loss figures are approximate."
        )
