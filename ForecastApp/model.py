import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from datetime import timedelta
from pathlib import Path


class ForecastModel:
    """Wrapper around SARIMAX to forecast sales or numeric series."""

    def __init__(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        self.order = order
        self.seasonal_order = seasonal_order

    def forecast(self, series: pd.Series, steps: int):
        model = SARIMAX(series, order=self.order, seasonal_order=self.seasonal_order)
        model_fit = model.fit(disp=False)
        prediction = model_fit.get_forecast(steps=steps)
        return prediction.predicted_mean, prediction.se_mean


TIME_MAP = {
    "next week": 7,
    "next month": 30,
    "next quarter": 90,
    "next Half Year": 182,
    "next year": 365,
}


def plot_forecast(history: pd.Series, forecast: pd.Series, errors: pd.Series, period: str, label: str, output_dir: Path) -> Path:
    last_date = history.index[-1]
    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=len(forecast), freq='D')
    plt.figure(figsize=(12, 6))
    plt.plot(history.index, history.values, label="Historical")
    plt.plot(forecast_dates, forecast.values, linestyle='--', marker='o', color='red', label=f"Forecast {period}")
    plt.fill_between(forecast_dates, forecast - 1.96*errors, forecast + 1.96*errors, color='red', alpha=0.2)
    plt.title(f"{label} Forecast")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"forecast_{label.replace(' ','_')}.png"
    plt.savefig(output_file)
    plt.close()
    return output_file
