"""Demand forecasting: SARIMA vs machine learning on a seasonal series.

Uses the classic monthly airline passengers series (1949-1960, strong
trend + yearly seasonality) to demonstrate the standard demand
forecasting workflow: explore trend/seasonality, test stationarity,
fit a seasonal ARIMA, compare against a lag-feature ML model and a
seasonal-naive baseline on a held-out final year.
"""
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

df = pd.read_csv("data/passengers.csv", parse_dates=["Month"], index_col="Month")
y = df["Passengers"]
print(f"Series: {len(y)} months, {y.index.min():%Y-%m} to {y.index.max():%Y-%m}")

# --- EDA: series + decomposition ---
y.plot(figsize=(11, 4), title="Monthly passengers (thousands)")
plt.tight_layout()
plt.savefig("plots/series.png", dpi=120)
plt.close()

decomp = seasonal_decompose(y, model="multiplicative", period=12)
fig = decomp.plot()
fig.set_size_inches(11, 8)
plt.tight_layout()
plt.savefig("plots/decomposition.png", dpi=120)
plt.close()
print("Saved series and decomposition plots")

# --- Stationarity: raw series should fail, differenced should pass ---
for label, series in [("raw", y), ("seasonally differenced", y.diff(12).dropna())]:
    p_value = adfuller(series)[1]
    print(f"ADF p-value ({label}): {p_value:.4f} "
          f"({'stationary' if p_value < 0.05 else 'not stationary'})")

# --- Hold out the final year ---
train, test = y.iloc[:-12], y.iloc[-12:]

def score(name, forecast):
    rmse = mean_squared_error(test, forecast) ** 0.5
    mape = np.mean(np.abs((test - forecast) / test)) * 100
    return name, rmse, mape

results = []

# Baseline: seasonal naive — repeat the last observed year
naive = train.iloc[-12:].values
results.append(score("Seasonal naive (baseline)", naive))

# SARIMA(1,1,1)(1,1,1,12): differencing handles trend + seasonality
sarima = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
sarima_forecast = sarima.forecast(steps=12)
results.append(score("SARIMA(1,1,1)(1,1,1,12)", sarima_forecast))

# ML: features all known at forecast time — lag-12, trend, month
def make_features(series):
    frame = pd.DataFrame({"y": series})
    frame["lag12"] = frame["y"].shift(12)
    frame["trend"] = np.arange(len(frame))
    frame["month"] = frame.index.month
    return frame.dropna()

features = make_features(y)
f_train = features[features.index <= train.index[-1]]
f_test = features[features.index > train.index[-1]]
ml = RandomForestRegressor(n_estimators=300, random_state=42)
ml.fit(f_train[["lag12", "trend", "month"]], f_train["y"])
ml_forecast = ml.predict(f_test[["lag12", "trend", "month"]])
results.append(score("Random forest (lag features)", ml_forecast))

# --- Report ---
lines = ["Demand forecasting — 12-month holdout (1960)\n"]
for name, rmse, mape in results:
    lines.append(f"{name}: RMSE {rmse:.1f}   MAPE {mape:.1f}%")
best = min(results, key=lambda r: r[1])
lines.append(f"\nBest by RMSE: {best[0]} ({best[1]:.1f})")
report = "\n".join(lines)
print("\n" + report)
Path("results.txt").write_text(report)

# --- Forecast plot ---
plt.figure(figsize=(11, 5))
plt.plot(y.index, y.values, label="Actual", color="black")
plt.plot(test.index, naive, label="Seasonal naive", linestyle="--")
plt.plot(test.index, sarima_forecast, label="SARIMA")
plt.plot(test.index, ml_forecast, label="Random forest")
plt.axvline(train.index[-1], color="gray", linestyle=":")
plt.title("Forecast vs actual — final year holdout")
plt.ylabel("Passengers (thousands)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/forecast_comparison.png", dpi=120)
plt.close()
print("Saved forecast plot to plots/")
