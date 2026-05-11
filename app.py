import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="ChatGPT Forecast Dashboard",
    layout="wide"
)

st.title("📈 ChatGPT Popularity Forecast Dashboard")

st.markdown("""
Interactive dashboard for analyzing and forecasting ChatGPT popularity using Time Series Forecasting Models.
""")

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("small_data.csv")

# IMPORTANT
# change column names if needed
# -----------------------------

# Example:
# df['date'] = pd.to_datetime(df['date'])

date_column = df.columns[0]
value_column = df.columns[1]

df[date_column] = pd.to_datetime(df[date_column])

df = df.sort_values(by=date_column)

df.set_index(date_column, inplace=True)

# =========================
# SIDEBAR
# =========================

st.sidebar.header("⚙ Model Parameters")

p = st.sidebar.slider("AR Order (p)", 0, 5, 2)
d = st.sidebar.slider("Differencing Order (d)", 0, 2, 1)
q = st.sidebar.slider("MA Order (q)", 0, 5, 1)

forecast_steps = st.sidebar.slider(
    "Forecast Steps",
    5,
    60,
    24
)

# =========================
# RAW DATA
# =========================

st.subheader("📊 Raw Time Series")

fig = px.line(
    df,
    y=value_column,
    title="ChatGPT Popularity Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# ADF TEST
# =========================

st.subheader("🧪 Stationarity Test (ADF)")

adf_result = adfuller(df[value_column])

adf_stat = adf_result[0]
p_value = adf_result[1]

col1, col2 = st.columns(2)

col1.metric("ADF Statistic", round(adf_stat, 4))
col2.metric("p-value", round(p_value, 4))

if p_value < 0.05:
    st.success("The series is stationary.")
else:
    st.warning("The series is NOT stationary.")

# =========================
# ACF & PACF
# =========================

st.subheader("📉 ACF and PACF Plots")

col1, col2 = st.columns(2)

with col1:
    fig_acf, ax = plt.subplots(figsize=(6,4))
    plot_acf(df[value_column], ax=ax)
    st.pyplot(fig_acf)

with col2:
    fig_pacf, ax = plt.subplots(figsize=(6,4))
    plot_pacf(df[value_column], ax=ax)
    st.pyplot(fig_pacf)

# =========================
# TRAIN TEST SPLIT
# =========================

train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]
test = df.iloc[train_size:]

# =========================
# MODEL TRAINING
# =========================

st.subheader("🤖 ARIMA Forecasting")

model = ARIMA(
    train[value_column],
    order=(p, d, q)
)

model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

# =========================
# METRICS
# =========================

mae = mean_absolute_error(test[value_column], forecast)

rmse = np.sqrt(
    mean_squared_error(test[value_column], forecast)
)

mape = np.mean(
    np.abs(
        (test[value_column] - forecast)
        / test[value_column]
    )
) * 100

# =========================
# METRICS DISPLAY
# =========================

st.subheader("📌 Model Evaluation")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", round(mae, 2))
c2.metric("RMSE", round(rmse, 2))
c3.metric("MAPE", round(mape, 2))

# =========================
# FORECAST PLOT
# =========================

st.subheader("📈 Forecast vs Actual")

forecast_df = pd.DataFrame({
    "Actual": test[value_column],
    "Forecast": forecast
})

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=train.index,
        y=train[value_column],
        name="Train"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test.index,
        y=test[value_column],
        name="Actual"
    )
)

fig2.add_trace(
    go.Scatter(
        x=test.index,
        y=forecast,
        name="Forecast"
    )
)

fig2.update_layout(
    title="ARIMA Forecast vs Actual",
    xaxis_title="Date",
    yaxis_title="Popularity"
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# FUTURE FORECAST
# =========================

st.subheader("🚀 Future Forecast")

future_forecast = model_fit.forecast(
    steps=forecast_steps
)

future_dates = pd.date_range(
    start=df.index[-1],
    periods=forecast_steps + 1,
    freq='D'
)[1:]

future_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_forecast
})

fig3 = go.Figure()

fig3.add_trace(
    go.Scatter(
        x=df.index,
        y=df[value_column],
        name="Historical Data"
    )
)

fig3.add_trace(
    go.Scatter(
        x=future_dates,
        y=future_forecast,
        name="Future Forecast"
    )
)

fig3.update_layout(
    title="Future Forecast of ChatGPT Popularity",
    xaxis_title="Date",
    yaxis_title="Popularity"
)

st.plotly_chart(fig3, use_container_width=True)

# =========================
# SHOW FORECAST TABLE
# =========================

st.subheader("📋 Forecast Values")

st.dataframe(future_df)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.markdown("""
Created using Streamlit • Time Series Forecasting Project
""")
Add Streamlit dashboard
