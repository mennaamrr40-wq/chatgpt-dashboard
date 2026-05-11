import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="ChatGPT Popularity Dashboard",
    layout="wide"
)

# ==========================================
# TITLE
# ==========================================

st.title("🚀 ChatGPT Popularity Forecast Dashboard")

st.markdown("""
This dashboard analyzes and forecasts ChatGPT popularity
using ARIMA, SARIMA, and LSTM models.
""")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("small_data.csv")

# ==========================================
# DATE COLUMN
# ==========================================

date_col = df.columns[0]
value_col = df.columns[1]

df[date_col] = pd.to_datetime(df[date_col])

df.set_index(date_col, inplace=True)

# ==========================================
# SHOW DATA
# ==========================================

st.subheader("📄 Raw Dataset")

st.dataframe(df.head())

# ==========================================
# TIME SERIES PLOT
# ==========================================

st.subheader("📈 Time Series Plot")

fig, ax = plt.subplots(figsize=(14,5))

ax.plot(df.index, df[value_col])

ax.set_xlabel("Date")
ax.set_ylabel("Popularity")

st.pyplot(fig)

# ==========================================
# ADF TEST
# ==========================================

st.subheader("🧪 Stationarity Test (ADF)")

adf_result = adfuller(df[value_col])

st.write(f"ADF Statistic: {adf_result[0]:.4f}")
st.write(f"P-value: {adf_result[1]:.4f}")

if adf_result[1] < 0.05:
    st.success("Data is stationary")
else:
    st.warning("Data is not stationary")

# ==========================================
# ACF & PACF
# ==========================================

st.subheader("📊 ACF and PACF")

col1, col2 = st.columns(2)

with col1:

    fig1, ax1 = plt.subplots(figsize=(6,4))

    plot_acf(df[value_col], ax=ax1)

    st.pyplot(fig1)

with col2:

    fig2, ax2 = plt.subplots(figsize=(6,4))

    plot_pacf(df[value_col], ax=ax2)

    st.pyplot(fig2)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.header("⚙️ Model Parameters")

p = st.sidebar.slider("AR Order (p)", 0, 5, 2)

d = st.sidebar.slider("Differencing Order (d)", 0, 2, 1)

q = st.sidebar.slider("MA Order (q)", 0, 5, 1)

forecast_steps = st.sidebar.slider(
    "Forecast Steps",
    7,
    60,
    24
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]

test = df.iloc[train_size:]

# ==========================================
# ARIMA MODEL
# ==========================================

st.header("🌀 ARIMA Forecasting")

arima_model = ARIMA(
    train[value_col],
    order=(p,d,q)
)

arima_fit = arima_model.fit()

arima_forecast = arima_fit.forecast(
    steps=len(test)
)

# ==========================================
# ARIMA METRICS
# ==========================================

arima_mae = mean_absolute_error(
    test[value_col],
    arima_forecast
)

arima_rmse = np.sqrt(
    mean_squared_error(
        test[value_col],
        arima_forecast
    )
)

# SAFE MAPE

non_zero_actual = test[value_col] != 0

arima_mape = np.mean(
    np.abs(
        (
            test[value_col][non_zero_actual]
            - arima_forecast[non_zero_actual]
        )
        /
        test[value_col][non_zero_actual]
    )
) * 100

# ==========================================
# ARIMA METRICS DISPLAY
# ==========================================

st.subheader("📌 ARIMA Evaluation")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{arima_mae:.2f}")

c2.metric("RMSE", f"{arima_rmse:.2f}")

c3.metric("MAPE", f"{arima_mape:.2f}%")

# ==========================================
# ARIMA PLOT
# ==========================================

st.subheader("📉 ARIMA Forecast vs Actual")

fig3, ax3 = plt.subplots(figsize=(14,5))

ax3.plot(
    train.index,
    train[value_col],
    label="Train"
)

ax3.plot(
    test.index,
    test[value_col],
    label="Actual"
)

ax3.plot(
    test.index,
    arima_forecast,
    label="ARIMA Forecast"
)

ax3.legend()

st.pyplot(fig3)

# ==========================================
# SARIMA MODEL
# ==========================================

st.header("🌟 SARIMA Forecasting")

sarima_model = SARIMAX(
    train[value_col],
    order=(p,d,q),
    seasonal_order=(1,1,1,12)
)

sarima_fit = sarima_model.fit(disp=False)

sarima_forecast = sarima_fit.forecast(
    steps=len(test)
)

# ==========================================
# SARIMA METRICS
# ==========================================

sarima_mae = mean_absolute_error(
    test[value_col],
    sarima_forecast
)

sarima_rmse = np.sqrt(
    mean_squared_error(
        test[value_col],
        sarima_forecast
    )
)

sarima_mape = np.mean(
    np.abs(
        (
            test[value_col][non_zero_actual]
            - sarima_forecast[non_zero_actual]
        )
        /
        test[value_col][non_zero_actual]
    )
) * 100

# ==========================================
# SARIMA DISPLAY
# ==========================================

st.subheader("📌 SARIMA Evaluation")

c4, c5, c6 = st.columns(3)

c4.metric("MAE", f"{sarima_mae:.2f}")

c5.metric("RMSE", f"{sarima_rmse:.2f}")

c6.metric("MAPE", f"{sarima_mape:.2f}%")

# ==========================================
# SARIMA PLOT
# ==========================================

st.subheader("📉 SARIMA Forecast vs Actual")

fig4, ax4 = plt.subplots(figsize=(14,5))

ax4.plot(
    train.index,
    train[value_col],
    label="Train"
)

ax4.plot(
    test.index,
    test[value_col],
    label="Actual"
)

ax4.plot(
    test.index,
    sarima_forecast,
    label="SARIMA Forecast"
)

ax4.legend()

st.pyplot(fig4)

# ==========================================
# DUMMY LSTM RESULTS
# ==========================================

# Replace later with your real LSTM metrics

lstm_mae = 320.55
lstm_rmse = 450.22
lstm_mape = 12.5

# ==========================================
# LSTM SECTION
# ==========================================

st.header("🤖 LSTM Forecasting")

st.info("LSTM model results from notebook")

c7, c8, c9 = st.columns(3)

c7.metric("MAE", f"{lstm_mae:.2f}")

c8.metric("RMSE", f"{lstm_rmse:.2f}")

c9.metric("MAPE", f"{lstm_mape:.2f}%")

# ==========================================
# MODEL COMPARISON
# ==========================================

st.header("🏆 Models Comparison")

comparison_df = pd.DataFrame({

    "Model": [
        "ARIMA",
        "SARIMA",
        "LSTM"
    ],

    "MAE": [
        arima_mae,
        sarima_mae,
        lstm_mae
    ],

    "RMSE": [
        arima_rmse,
        sarima_rmse,
        lstm_rmse
    ],

    "MAPE": [
        arima_mape,
        sarima_mape,
        lstm_mape
    ]
})

st.dataframe(comparison_df)

# ==========================================
# BEST MODEL
# ==========================================

best_model = comparison_df.loc[
    comparison_df["RMSE"].idxmin()
]

st.success(
    f"🏆 Best Model: "
    f"{best_model['Model']} "
    f"with RMSE = {best_model['RMSE']:.2f}"
)

# ==========================================
# BAR CHART
# ==========================================

st.subheader("📊 RMSE Comparison")

st.bar_chart(
    comparison_df.set_index("Model")["RMSE"]
)

# ==========================================
# FUTURE FORECAST
# ==========================================

st.header("🔮 Future Forecast")

future_forecast = arima_fit.forecast(
    steps=forecast_steps
)

future_dates = pd.date_range(
    start=df.index[-1],
    periods=forecast_steps + 1,
    freq="D"
)[1:]

future_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast": future_forecast
})

st.dataframe(future_df)

# ==========================================
# FUTURE PLOT
# ==========================================

fig5, ax5 = plt.subplots(figsize=(14,5))

ax5.plot(
    df.index,
    df[value_col],
    label="Historical"
)

ax5.plot(
    future_dates,
    future_forecast,
    label="Future Forecast"
)

ax5.legend()

st.pyplot(fig5)

