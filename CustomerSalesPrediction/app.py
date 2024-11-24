import pandas as pd
import matplotlib
matplotlib.use('Agg') ### Use the Agg for backend for non-interactive plotting
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
from flask import Flask, render_template, request
from datetime import time, timedelta,datetime
import os
import webbrowser
import time as time_module ###Import time as a different name


app = Flask(__name__)

# Load the Excel file
file_path = r'CustomerSalesPrediction\DataSet\Training Data.xlsx'

"""Function to load and prepare data"""
def load_data():
    try:
        df = pd.read_excel(file_path, engine='openpyxl', usecols=['Customercode', 'Reportdate', 'SKU', 'SalesUnits'])
        df['Reportdate'] = pd.to_datetime(df['Reportdate'], errors='coerce')
        df = df.dropna(subset=['Reportdate'])
        grouped_data = df.groupby(['Customercode', 'SKU', 'Reportdate'])['SalesUnits'].sum().reset_index()
        print("Data loaded successfully.")
        return grouped_data
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

"""Function to forecast future sales SARIMA"""
def forecast_sales(sales_series, steps):
    try:
        model = SARIMAX(sales_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.get_forecast(steps=steps)
        return forecast.predicted_mean, forecast.se_mean  # Return both predicted means and standard errors
    except Exception as e:
        print(f"Error in SARIMA model fitting: {e}")
        return np.zeros(steps), np.zeros(steps)  # Return zeros in case of error

@app.route('/')
def index():
    version = int(time_module.time()) 
    return render_template('index.html', version=version)

@app.route('/forecast', methods=['POST'])
def forecast():
    customer_code = request.form.get('customer_code').strip()
    grouped_data = load_data()

    if grouped_data is None:
        return "Error loading data."

    print(f"Loaded Data:\n{grouped_data.head()}")  # Debug print

    customers = grouped_data['Customercode'].unique()

    if customer_code not in customers and customer_code.lower() != 'all':
        return "Invalid Customer Code. Please enter a valid code."

    if customer_code.lower() == 'all':
        all_skus = {customer: grouped_data[grouped_data['Customercode'] == customer]['SKU'].unique() for customer in customers}
        return render_template('results.html', all_skus=all_skus)

    customer_data = grouped_data[grouped_data['Customercode'] == customer_code]
    skus = customer_data['SKU'].unique()

    sku = request.form.get('sku').strip()

    print(f"Customer Code: {customer_code}, SKU: {sku}")  # Debug print

    if sku.lower() == 'all':
        results = []
        for sku in skus:
            sku_data = customer_data[customer_data['SKU'] == sku].sort_values(by='Reportdate')
            time_series = sku_data.set_index('Reportdate')['SalesUnits']
            forecast_mean, forecast_se = forecast_sales(time_series, 6)

            print(f"Forecast Mean for SKU {sku}: {forecast_mean}")  # Debug print

            if np.all(forecast_mean == 0):
                return f"Forecasting failed for SKU {sku} due to insufficient data."

            last_date = time_series.index[-1]
            forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=6, freq='ME')  

            '''Plotting'''
            plt.figure(figsize=(12, 6))
            plt.bar(sku_data['Reportdate'], sku_data['SalesUnits'], color='skyblue', label=f'Actual Sales - Customer {customer_code}, SKU {sku}')
            plt.plot(forecast_dates, forecast_mean, marker='o', linestyle='--', color='red', label=f'Forecasted Sales (Next 6 months) - SKU {sku}')
            plt.fill_between(forecast_dates, forecast_mean - 1.96 * forecast_se, forecast_mean + 1.96 * forecast_se, color='red', alpha=0.2)
            plt.title(f'Sales Units for Customer {customer_code} and SKU {sku} (with Forecast)')
            plt.xlabel('Reported Date')
            plt.ylabel('Sum of Sales Units')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            plt.gcf().autofmt_xdate()
            plt.legend()
            current_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # Example format
            plot_filename = f'CustomerSalesPrediction/Outputs/forecast_{customer_code}_{sku}_{current_date}.png'
            plt.savefig(plot_filename)
            plt.close()
            print(f"Plot saved at: {plot_filename}") 
            results.append((sku, plot_filename))
        return render_template('results.html', results=results)

    else:
        if sku in skus:
            sku_data = customer_data[customer_data['SKU'] == sku].sort_values(by='Reportdate')
            time_series = sku_data.set_index('Reportdate')['SalesUnits']
            forecast_mean, forecast_se = forecast_sales(time_series, 6)

            print(f"Forecast Mean for SKU {sku}: {forecast_mean}")  # Debug print

            if np.all(forecast_mean == 0):
                return f"Forecasting failed for SKU {sku} due to insufficient data."

            last_date = time_series.index[-1]
            forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=6, freq='ME')  # Updated frequency

            '''Plotting'''
            plt.figure(figsize=(12, 6))
            plt.bar(sku_data['Reportdate'], sku_data['SalesUnits'], color='blue', label=f'Actual Sales - Customer {customer_code}, SKU {sku}')
            plt.plot(forecast_dates, forecast_mean, marker='o', linestyle='--', color='red', label=f'Forecasted Sales (Next 6 months) - SKU {sku}')
            plt.fill_between(forecast_dates, forecast_mean - 1.96 * forecast_se, forecast_mean + 1.96 * forecast_se, color='red', alpha=0.2)
            plt.title(f'Sales Units for Customer {customer_code} and SKU {sku} (with Forecast)')
            plt.xlabel('Reported Date')
            plt.ylabel('Sum of Sales Units')
            plt.grid(True)
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            plt.gcf().autofmt_xdate()
            plt.legend()
            current_date = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')  # Example format
            plot_filename = f'CustomerSalesPrediction/Outputs/forecast_{customer_code}_{sku}_{current_date}.png'
            plt.savefig(plot_filename)
            plt.close()
            print(f"Plot saved at: {plot_filename}") 
            return render_template('results.html', results=[(sku, plot_filename)])
        else:
            return f"SKU '{sku}' not found for Customer '{customer_code}'."

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000")
    app.run(debug=True)
