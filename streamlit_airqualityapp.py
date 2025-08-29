# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import altair as alt

# Mavjud yordamchi funksiyalar
from functions import air_quality, get_weather, add_time_features

st.set_page_config(
    page_title="Uzbekistan Air Quality & Storm Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------
# Auto-refresh (every 30 minutes)
# ------------------------
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = datetime.utcnow()

now = datetime.utcnow()
if (now - st.session_state["last_refresh"]).total_seconds() > 1800:  # 1800s = 30min
    st.session_state["last_refresh"] = now
    st.rerun()

# ------------------------
# Helper utilities
# ------------------------
@st.cache_data(ttl=600)
def load_model(path="best_model.pkl"):
    try:
        model = joblib.load(path)
        return model, None
    except Exception as e:
        return None, str(e)

def prepare_forecast_for_model(forecast_df, features):
    forecast_df = forecast_df.copy()
    forecast_df['date'] = pd.to_datetime(forecast_df['date'], utc=True, errors='coerce')
    forecast_feat = add_time_features(forecast_df, 'date')
    for col in ['value_lag1','value_lag24','value_roll6','value_roll24']:
        if col not in forecast_feat.columns:
            forecast_feat[col] = np.nan
    X_forecast = forecast_feat[[c for c in features if c in forecast_feat.columns]]
    return forecast_df, forecast_feat, X_forecast

def compute_disaster_warnings(df):
    warnings = []
    dfp = df.set_index('date').sort_index()

    # Pressure drop
    if 'surface_pressure' in df.columns and len(dfp) >= 4:
        dfp['p_3h_diff'] = dfp['surface_pressure'] - dfp['surface_pressure'].shift(3)
        recent = dfp['p_3h_diff'].iloc[-1]
        if recent <= -6:
            warnings.append("⚠️ Pressure dropped ≥6 hPa in last 3h → possible storm front 🌪")

    # Strong winds
    if 'windspeed' in df.columns:
        max_w = df['windspeed'].max()
        if max_w >= 15:
            warnings.append("💨 Strong wind forecast (≥15 m/s) → storm risk")

    # Heavy rain
    if 'precipitation' in df.columns and len(dfp) >= 24:
        tot24 = dfp['precipitation'].rolling(24, min_periods=1).sum().iloc[-1]
        if tot24 >= 20:
            warnings.append("🌧 Heavy rainfall (≥20 mm/24h) → flooding risk")

    # Sudden temperature drop
    if 'temperature_2m' in df.columns and len(dfp) >= 6:
        temp_diff = dfp['temperature_2m'].iloc[-1] - dfp['temperature_2m'].iloc[-6]
        if temp_diff <= -5:
            warnings.append(f"🌡 Sudden temp drop ({temp_diff:.1f}°C in 6h) → storm front signal")

    # Heatwave
    if 'temperature_2m' in df.columns and df['temperature_2m'].max() > 40:
        warnings.append("🔥 Extreme heat (>40°C) → heatwave risk")

    # Humidity + heat stress
    if 'humidity' in df.columns and 'temperature_2m' in df.columns:
        if df['humidity'].iloc[-1] > 85 and df['temperature_2m'].iloc[-1] > 30:
            warnings.append("🥵 High humidity + heat → heat stress risk")

    # Icing / snowfall
    if 'temperature_2m' in df.columns and 'precipitation' in df.columns:
        if df['temperature_2m'].min() < 0 and df['precipitation'].sum() > 0:
            warnings.append("❄️ Subzero + precipitation → icing/snowfall risk")

    # Fog risk
    if 'cloudcover' in df.columns and 'humidity' in df.columns:
        if df['cloudcover'].iloc[-1] > 90 and df['humidity'].iloc[-1] > 80:
            warnings.append("🌫 High humidity + cloudcover → fog risk")

    return warnings

# Features list
FEATURES = [
    'temperature_2m','apparent_temperature','precipitation','rain','snowfall',
    'humidity','surface_pressure','year','month','day','hour','dayofweek',
    'is_weekend','hour_sin','hour_cos','dow_sin','dow_cos',
    'value_lag1','value_lag24','value_roll6','value_roll24'
]

# ------------------------
# Sidebar (read-only)
# ------------------------
st.sidebar.title("Controls (fixed)")
st.sidebar.text("Data source: OpenAQ + OpenMeteo")
st.sidebar.text("Forecast update: every 30 minutes")
st.sidebar.text("Forecast days: 3 (default)")
st.sidebar.text("Model: best_model.pkl")
st.sidebar.text("Predictions saved to CSV")

# ------------------------
# Title
# ------------------------
st.title("🇺🇿 Uzbekistan — Air Quality & Weather Forecast")
st.markdown("Machine learning system: **PM2.5 prediction** + weather dashboard + storm warnings.")

model, err = load_model("best_model.pkl")

# ------------------------
# Data
# ------------------------
st.header("1) Data")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Realtime Air Quality")
    try:
        aq_df = air_quality()
        if not aq_df.empty:
            row = aq_df.iloc[0]
            st.write("📍 Location: Tashkent (Chilanzar)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("CO (mg/m³)", row["co"])
                st.metric("SO₂ (µg/m³)", row["so2"])
                st.metric("Humidity (%)", row["h"])
            with c2:
                st.metric("O₃ (µg/m³)", row["o3"])
                st.metric("Pressure (hPa)", row["p"])
                st.metric("Temperature (°C)", row["t"])
            with c3:
                st.metric("Wind Speed (m/s)", row["w"])
                st.metric("Wind Gust (m/s)", row["wg"])
                st.metric("Dew Point (°C)", row["dew"])
        else:
            st.warning("No realtime data available.")
    except Exception as e:
        st.error(f"Realtime error: {e}")

with col_b:
    st.subheader("Forecast (weather)")
    try:
        _, forecast_df = get_weather(date_from=datetime.utcnow().date(),
                                     date_till=(datetime.utcnow().date() + timedelta(days=3)),
                                     chunk_days=7)
        st.success("✅ Forecast fetched")
        st.dataframe(forecast_df.head(10))
    except Exception as e:
        st.error(f"Forecast fetch failed: {e}")
        forecast_df = pd.DataFrame()

# ------------------------
# Prediction
# ------------------------
st.header("2) Prediction & Graphs")

if model is not None and not forecast_df.empty:
    forecast_df_prepared, forecast_feat, X_forecast = prepare_forecast_for_model(forecast_df, FEATURES)
    try:
        preds = model.predict(X_forecast)
        forecast_df_prepared['prediction'] = preds
        st.success("✅ Predictions done")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("PM2.5 Forecast (next hours)")
    pm_chart = alt.Chart(forecast_df_prepared).mark_line(color="red").encode(
        x="date:T", y=alt.Y("prediction:Q", title="PM2.5 (µg/m³)"),
        tooltip=["date:T","prediction:Q"]
    ).properties(width=800, height=300)
    st.altair_chart(pm_chart, use_container_width=True)

    st.subheader("Weather Features")
    weather_cols = ["temperature_2m","humidity","windspeed","surface_pressure","precipitation","cloudcover"]
    df_melted = forecast_df_prepared.melt(id_vars=["date"], value_vars=weather_cols, 
                                          var_name="feature", value_name="value")
    weather_chart = alt.Chart(df_melted).mark_line().encode(
        x="date:T", y="value:Q", color="feature:N",
        tooltip=["date:T","feature:N","value:Q"]
    ).properties(width=800, height=400)
    st.altair_chart(weather_chart, use_container_width=True)

    st.subheader("Warnings & System Flags")
    warnings = compute_disaster_warnings(forecast_df_prepared)
    if warnings:
        for w in warnings:
            if "storm" in w.lower():
                st.error("🔴 " + w)
            elif "wind" in w.lower():
                st.warning("🟠 " + w)
            elif "rain" in w.lower():
                st.warning("🟡 " + w)
            else:
                st.info("🟢 " + w)
    else:
        st.success("🟢 Stable — no immediate disaster warnings.")

# ------------------------
# Health advice
# ------------------------
st.header("3) Health Advice (based on PM2.5)")
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

st.markdown("---")
st.caption("Built for Uzbekistan (Central Asia) — PM2.5 + weather forecasting & storm warnings.")
