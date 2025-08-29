import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta, timezone
import altair as alt

# -----------------------------
# Helper funksiyalar
# -----------------------------
from functions import air_quality, get_weather, add_time_features

# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="Uzbekistan Air Quality & Storm Prediction",
    layout="wide",
)
st.markdown("This app shows **air quality (PM2.5)** and **weather forecast** with simple storm warnings.")

# -----------------------------
# Auto-refresh (30 min)
# -----------------------------
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = datetime.now(timezone.utc)

now = datetime.now(timezone.utc)
if (now - st.session_state["last_refresh"]) > timedelta(minutes=30):
    st.session_state["last_refresh"] = now
    st.rerun()

# -----------------------------
# Model loader
# -----------------------------
@st.cache_data(ttl=600)
def load_model(path="best_model.pkl"):
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Forecast prep
# -----------------------------
def prepare_forecast_for_model(forecast_df, features):
    forecast_df = forecast_df.copy()
    forecast_df['date'] = pd.to_datetime(forecast_df['date'], utc=True, errors='coerce')
    forecast_feat = add_time_features(forecast_df, 'date')
    for col in ['value_lag1','value_lag24','value_roll6','value_roll24']:
        if col not in forecast_feat.columns:
            forecast_feat[col] = np.nan
    X_forecast = forecast_feat[[c for c in features if c in forecast_feat.columns]]
    return forecast_df, forecast_feat, X_forecast

# -----------------------------
# Disaster warning rules
# -----------------------------
def compute_disaster_warnings(df):
    warnings = []
    dfp = df.set_index('date').sort_index()

    if 'surface_pressure' in df.columns and len(dfp) >= 4:
        dfp['p_3h_diff'] = dfp['surface_pressure'] - dfp['surface_pressure'].shift(3)
        recent = dfp['p_3h_diff'].iloc[-1]
        if recent <= -6:
            warnings.append("⚠️ Pressure dropped ≥6 hPa in last 3h → possible storm front 🌪")

    if 'windspeed' in df.columns:
        max_w = df['windspeed'].max()
        if max_w >= 15:
            warnings.append("💨 Strong wind forecast (≥15 m/s) → storm risk")

    if 'precipitation' in df.columns and len(dfp) >= 24:
        tot24 = dfp['precipitation'].rolling(24, min_periods=1).sum().iloc[-1]
        if tot24 >= 20:
            warnings.append("🌧 Heavy rainfall (≥20 mm/24h) → flooding risk")

    if 'temperature_2m' in df.columns and len(dfp) >= 6:
        temp_diff = dfp['temperature_2m'].iloc[-1] - dfp['temperature_2m'].iloc[-6]
        if temp_diff <= -5:
            warnings.append(f"🌡 Sudden temp drop ({temp_diff:.1f}°C in 6h) → storm front signal")

    if 'temperature_2m' in df.columns:
        if df['temperature_2m'].max() > 40:
            warnings.append("🔥 Extreme heat (>40°C) → heatwave risk")

    if 'humidity' in df.columns and 'temperature_2m' in df.columns:
        last_hum = df['humidity'].iloc[-1]
        last_temp = df['temperature_2m'].iloc[-1]
        if last_hum > 85 and last_temp > 30:
            warnings.append("🥵 High humidity + heat → heat stress risk")

    if 'temperature_2m' in df.columns and 'precipitation' in df.columns:
        if df['temperature_2m'].min() < 0 and df['precipitation'].sum() > 0:
            warnings.append("❄️ Subzero + precipitation → icing/snowfall risk")

    if 'cloudcover' in df.columns and 'humidity' in df.columns:
        last_cc = df['cloudcover'].iloc[-1]
        last_h = df['humidity'].iloc[-1]
        if last_cc > 90 and last_h > 80:
            warnings.append("🌫 High humidity + cloudcover → fog risk")

    return warnings

# -----------------------------
# Features
# -----------------------------
FEATURES = [
    'temperature_2m','apparent_temperature','precipitation','rain','snowfall',
    'humidity','surface_pressure','year','month','day','hour','dayofweek',
    'is_weekend','hour_sin','hour_cos','dow_sin','dow_cos',
    'value_lag1','value_lag24','value_roll6','value_roll24'
]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.text("Data: OpenAQ + OpenMeteo")
st.sidebar.text("Refresh: every 30 minutes")
st.sidebar.text("Forecast: 3 days")
st.sidebar.text("Model: best_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🇺🇿 Uzbekistan — Air Quality & Weather Forecast")

# -----------------------------
# Data
# -----------------------------
st.header("1) Data")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Realtime Air Quality")
    try:
        aq_df = air_quality()
        if not aq_df.empty:
            row = aq_df.iloc[0]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Temperature (°C)", row["t"])
                st.metric("Humidity (%)", row["h"])
            with col2:
                st.metric("Pressure (hPa)", row["p"])
                st.metric("CO (mg/m³)", row["co"])
            with col3:
                st.metric("Wind Speed (m/s)", row["w"])
                st.metric("SO₂ (µg/m³)", row["so2"])
        else:
            st.warning("No realtime data available.")
    except Exception as e:
        st.error(f"Realtime air quality error: {e}")

with col_b:
    st.subheader("Weather Forecast (3 days)")
    try:
        today = datetime.now(timezone.utc).date()
        _, forecast_df = get_weather(date_from=today,
                                     date_till=(today + timedelta(days=3)),
                                     chunk_days=7)
        st.success("✅ Forecast fetched")
        st.dataframe(forecast_df.head(10))
    except Exception as e:
        st.error(f"Forecast fetch failed: {e}")
        forecast_df = pd.DataFrame()

# -----------------------------
# Prediction
# -----------------------------
st.header("2) Prediction & Graphs")

model, err = load_model("best_model.pkl")

if model is None or forecast_df.empty:
    st.info("Model or forecast data missing — cannot run prediction.")
else:
    forecast_df_prepared, forecast_feat, X_forecast = prepare_forecast_for_model(forecast_df, FEATURES)
    try:
        preds = model.predict(X_forecast)
        forecast_df_prepared['prediction'] = preds
        st.success("✅ Predictions done")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # PM2.5 forecast
    st.subheader("PM2.5 Forecast (next hours)")
    df_pm = forecast_df_prepared.dropna(subset=["prediction"])  # NaN larni olib tashlash
    if not df_pm.empty:
        pm_chart = alt.Chart(df_pm).mark_line(color="red").encode(
            x="date:T",
            y=alt.Y("prediction:Q",
                    title="PM2.5 (µg/m³)",
                    scale=alt.Scale(zero=False)),  # nolni o‘chirib tashlash
            tooltip=["date:T","prediction:Q"]
        ).properties(width=800, height=300)
        st.altair_chart(pm_chart, use_container_width=True)
    else:
        st.warning("No PM2.5 prediction data available.")

    # Weather features split
    st.subheader("Weather Features (Temperature, Humidity, Wind, Cloudcover)")
    weather_cols_main = ["temperature_2m","humidity","windspeed","cloudcover"]
    df_main = forecast_df_prepared.melt(id_vars=["date"], value_vars=weather_cols_main,
                                        var_name="feature", value_name="value")
    chart_main = alt.Chart(df_main).mark_line().encode(
        x="date:T", y="value:Q", color="feature:N",
        tooltip=["date:T","feature:N","value:Q"]
    ).properties(width=800, height=300)
    st.altair_chart(chart_main, use_container_width=True)

    st.subheader("Precipitation (Rainfall in mm)")
    df_prec = forecast_df_prepared.melt(id_vars=["date"], value_vars=["precipitation"],
                                        var_name="feature", value_name="value")
    chart_prec = alt.Chart(df_prec).mark_bar(color="blue").encode(
        x="date:T", y="value:Q",
        tooltip=["date:T","value:Q"]
    ).properties(width=800, height=300)
    st.altair_chart(chart_prec, use_container_width=True)

    st.subheader("Surface Pressure (hPa)")
    df_press = forecast_df_prepared.melt(id_vars=["date"], value_vars=["surface_pressure"],
                                         var_name="feature", value_name="value")

    min_val = df_press["value"].min()
    max_val = df_press["value"].max()

    chart_press = alt.Chart(df_press).mark_line(color="green").encode(
        x="date:T",
        y=alt.Y("value:Q",
                title="Surface Pressure (hPa)",
                scale=alt.Scale(domain=[min_val - 5, max_val + 5], zero=False)),
        tooltip=["date:T","value:Q"]
    ).properties(width=800, height=300)

    st.altair_chart(chart_press, use_container_width=True)

    # Warnings
    st.subheader("⚠️ Warnings")
    warnings = compute_disaster_warnings(forecast_df_prepared)
    if warnings:
        for w in warnings:
            if "storm" in w.lower() or "heat" in w.lower():
                st.error("🔴 " + w)
            elif "wind" in w.lower() or "rain" in w.lower():
                st.warning("🟠 " + w)
            else:
                st.info("🟢 " + w)
    else:
        st.success("🟢 Stable — no immediate disaster warnings.")

# -----------------------------
# Health advice
# -----------------------------
st.header("3) Health Advice (PM2.5)")
if 'forecast_df_prepared' in locals() and 'prediction' in forecast_df_prepared.columns:
    max_pred = forecast_df_prepared['prediction'].max()
    if max_pred > 150:
        st.error("🔴 Very Unhealthy — 😷 Avoid outdoors, wear N95.")
    elif max_pred > 100:
        st.warning("🟠 Unhealthy — 😷 Wear a mask, avoid long outdoor activity.")
    elif max_pred > 50:
        st.info("🟡 Moderate — Sensitive groups should limit outdoor exposure.")
    else:
        st.success("🟢 Safe — Air quality is good.")
else:
    st.info("No prediction to assess health advice.")

# -----------------------------
# Glossary
# -----------------------------
st.header("4) Glossary (Key Terms)")

glossary_items = {
    "🌫 PM2.5": "Tiny dust particles that can harm lungs.",
    "💧 Humidity (%)": "Amount of water vapor in the air.",
    "📉 Pressure (hPa)": "Air pressure; sudden drop → storm risk.",
    "🌬 Wind speed (m/s)": "How strong the wind blows.",
    "🌧 Rainfall (mm)": "Volume of rain; clears pollution but may flood.",
    "☁️ Cloud cover (%)": "How much of the sky is covered by clouds."
}

for icon, desc in glossary_items.items():
    st.markdown(f"<div style='background:#f8f9fa;padding:10px;border-radius:8px;margin-bottom:5px;'><b>{icon}</b> {desc}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Built for Uzbekistan (Central Asia) — PM2.5 + weather forecasting & storm warnings.")
