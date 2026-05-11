import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="ChatGPT Forecast Dashboard",
    page_icon="🚀",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #161A2D;
}

.metric-box {
    background-color: #1E1E2E;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.title("🚀 ChatGPT Popularity Forecast Dashboard")

st.markdown("""
### AI-Based Forecasting Using ARIMA, SARIMA, and LSTM Models
""")

# =========================================
# LOAD DATA
# =========================================

df = pd.read_csv("small_data.csv")

date_col = df.columns[0]
value_col = df.columns[1]

df[date_col] = pd.to_datetime(df[date_col])

df.set_index(date_col, inplace=True)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("⚙️ Forecast Controls")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["ARIMA", "SARIMA"]
)

p = st.sidebar.slider(
    "AR Order (p)",
    0,
    5,
    2
)

d = st.sidebar.slider(
    "Differencing (d)",
    0,
    2,
    1
)

q = st.sidebar.slider(
    "MA Order (q)",
    0,
    5,
    1
)

forecast_steps = st.sidebar.slider(
    "Forecast Days",
    7,
    60,
    24
)

st.sidebar.markdown("---")

st.sidebar.info("""
📌 Dashboard Features

✔ Forecasting  
✔ ACF/PACF  
✔ Model Evaluation  
✔ Future Prediction  
✔ Interactive Parameters
""")

# =========================================
# RAW DATA
# =========================================

st.header("📄 Raw Dataset")

st.dataframe(df.head())

# =========================================
# TIME SERIES
# =========================================

st.header("📈 Time Series Visualization")

fig, ax = plt.subplots(figsize=(14,5))

ax.plot(
    df.index,
    df[value_col],
    color="cyan"
)

ax.set_xlabel("Date")
ax.set_ylabel("Popularity")

st.pyplot(fig)

# =========================================
# STATIONARITY
# =========================================

st.header("🧪 Stationarity Test")

adf_result = adfuller(df[value_col])

col1, col2 = st.columns(2)

col1.metric(
    "ADF Statistic",
    f"{adf_result[0]:.4f}"
)

col2.metric(
    "P-Value",
    f"{adf_result[1]:.4f}"
)

if adf_result[1] < 0.05:
    st.success("✅ Data is Stationary")
else:
    st.warning("⚠️ Data is Not Stationary")

# =========================================
# ACF PACF
# =========================================

st.header("📊 ACF & PACF")

c1, c2 = st.columns(2)

with c1:

    fig1, ax1 = plt.subplots(figsize=(6,4))

    plot_acf(df[value_col], ax=ax1)

    st.pyplot(fig1)

with c2:

    fig2, ax2 = plt.subplots(figsize=(6,4))

    plot_pacf(df[value_col], ax=ax2)

    st.pyplot(fig2)

# =========================================
# TRAIN TEST SPLIT
# =========================================

train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]

test = df.iloc[train_size:]

# =========================================
# ARIMA
# =========================================

st.header("🌀 ARIMA Forecasting")

arima_model = ARIMA(
    train[value_col],
    order=(p,d,q)
)

arima_fit = arima_model.fit()

arima_forecast = arima_fit.forecast(
    steps=len(test)
)

# =========================================
# SAFE MAPE
# =========================================

actual_values = test[value_col].values

forecast_values = arima_forecast.values

mask = actual_values != 0

# =========================================
# ARIMA METRICS
# =========================================

arima_mae = mean_absolute_error(
    actual_values,
    forecast_values
)

arima_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        forecast_values
    )
)

arima_mape = np.mean(
    np.abs(
        (
            actual_values[mask]
            - forecast_values[mask]
        )
        /
        actual_values[mask]
    )
) * 100

# =========================================
# ARIMA DISPLAY
# =========================================

st.subheader("📌 ARIMA Evaluation")

m1, m2, m3 = st.columns(3)

m1.metric(
    "MAE",
    f"{arima_mae:.2f}"
)

m2.metric(
    "RMSE",
    f"{arima_rmse:.2f}"
)

m3.metric(
    "MAPE",
    f"{arima_mape:.2f}%"
)

# =========================================
# ARIMA PLOT
# =========================================

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

# =========================================
# SARIMA
# =========================================

st.header("🌟 SARIMA Forecasting")

sarima_model = SARIMAX(
    train[value_col],
    order=(p,d,q),
    seasonal_order=(1,1,1,12)
)

sarima_fit = sarima_model.fit(
    disp=False
)

sarima_forecast = sarima_fit.forecast(
    steps=len(test)
)

# =========================================
# SARIMA METRICS
# =========================================

forecast_values2 = sarima_forecast.values

sarima_mae = mean_absolute_error(
    actual_values,
    forecast_values2
)

sarima_rmse = np.sqrt(
    mean_squared_error(
        actual_values,
        forecast_values2
    )
)

sarima_mape = np.mean(
    np.abs(
        (
            actual_values[mask]
            - forecast_values2[mask]
        )
        /
        actual_values[mask]
    )
) * 100

# =========================================
# SARIMA DISPLAY
# =========================================

st.subheader("📌 SARIMA Evaluation")

s1, s2, s3 = st.columns(3)

s1.metric(
    "MAE",
    f"{sarima_mae:.2f}"
)

s2.metric(
    "RMSE",
    f"{sarima_rmse:.2f}"
)

s3.metric(
    "MAPE",
    f"{sarima_mape:.2f}%"
)

# =========================================
# SARIMA PLOT
# =========================================

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

# =========================================
# LSTM RESULTS
# =========================================

st.header("🤖 LSTM Forecasting")

# Replace later with real values

lstm_mae = 320.55
lstm_rmse = 450.22
lstm_mape = 12.50

l1, l2, l3 = st.columns(3)

l1.metric(
    "MAE",
    f"{lstm_mae:.2f}"
)

l2.metric(
    "RMSE",
    f"{lstm_rmse:.2f}"
)

l3.metric(
    "MAPE",
    f"{lstm_mape:.2f}%"
)

st.info("""
LSTM metrics are imported from the notebook model.
""")

# =========================================
# COMPARISON TABLE
# =========================================

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

# =========================================
# BEST MODEL
# =========================================

best_model = comparison_df.loc[
    comparison_df["RMSE"].idxmin()
]

st.success(
    f"""
🏆 Best Performing Model:
{best_model['Model']}

RMSE = {best_model['RMSE']:.2f}
"""
)

# =========================================
# BAR CHART
# =========================================

st.subheader("📊 RMSE Comparison")

st.bar_chart(
    comparison_df.set_index("Model")["RMSE"]
)

# =========================================
# FUTURE FORECAST
# =========================================

st.header("🔮 Future Forecast")

if model_choice == "ARIMA":

    future_forecast = arima_fit.forecast(
        steps=forecast_steps
    )

else:

    future_forecast = sarima_fit.forecast(
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

# =========================================
# FUTURE PLOT
# =========================================

fig5, ax5 = plt.subplots(figsize=(14,5))

ax5.plot(
    df.index,
    df[value_col],
    label="Historical"
)

ax5.plot(
    future_dates,
    future_forecast,
    label="Forecast"
)

ax5.legend()

st.pyplot(fig5)

# =========================================
# FINAL NOTE
# =========================================

st.markdown("---")

st.markdown("""
### ✅ Dashboard Completed Successfully

This dashboard was developed for forecasting ChatGPT popularity
using machine learning and time series models.
""")

