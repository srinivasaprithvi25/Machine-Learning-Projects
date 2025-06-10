# Machine Learning Projects

This repository showcases a simple sales forecasting application built with **Flask** and **SARIMAX**. The main project directory is [`CustomerSalesPrediction`](./CustomerSalesPrediction/) which contains the application code and sample data.

## Project Structure

- `CustomerSalesPrediction/app.py` – Flask application that loads sales data and generates forecasts using a SARIMA model.
- `CustomerSalesPrediction/DataSet/` – Directory where the training data is expected. A small CSV example is provided, but the code expects an Excel file named `Training Data.xlsx`.
- `CustomerSalesPrediction/static/` – Contains a small stylesheet.
- `CustomerSalesPrediction/templates/` – HTML templates for the web interface.
- `CustomerSalesPrediction/Outputs/` – Generated plots are saved here when the app runs (directory will be created automatically).

## Setup

1. Install Python 3 and the required packages:
   ```bash
   pip install flask pandas matplotlib numpy statsmodels openpyxl
   ```
2. Place your sales dataset in `CustomerSalesPrediction/DataSet/Training Data.xlsx` or adjust the path inside `app.py`.

## Running

Launch the Flask application from the project root:

```bash
python CustomerSalesPrediction/app.py
```

The application opens in your browser and lets you input a customer code and SKU to produce a six‑month forecast. Forecast images are stored in `CustomerSalesPrediction/Outputs/`.

## Notes

- The provided CSV in `DataSet` is a sample; depending on your data you may need to modify the file path or column names.
- SARIMA parameters in `app.py` (`order` and `seasonal_order`) are set to simple defaults and might need tuning for real-world datasets.

