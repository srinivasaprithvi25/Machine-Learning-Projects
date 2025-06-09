import os
import argparse
import pandas as pd
import openai
from pathlib import Path
from db_connection import get_db_engine
from model import ForecastModel, TIME_MAP, plot_forecast


def get_columns(prediction_type: str, prompt_file: Path) -> str:
    """Use OpenAI to determine columns needed for prediction."""
    prompt = prompt_file.read_text()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    message = f"Which columns are required to predict {prediction_type}?"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": message}],
        temperature=0.0,
    )
    return response.choices[0].message["content"].strip()


def build_query(columns: str, prediction_type: str, customer_code: str | None) -> str:
    base_table = '"SOPMT"."Synthetic_O2C_SO_header"'
    where_clause = f" WHERE customer_number = '{customer_code}'" if customer_code else ""
    return f"SELECT {columns} FROM {base_table}{where_clause}"


def main():
    parser = argparse.ArgumentParser(description="Business forecasting tool")
    parser.add_argument("prediction_type", choices=["Cash Flow Prediction", "Sales Prediction", "Inventory Prediction"], help="Type of prediction")
    parser.add_argument("time_period", choices=list(TIME_MAP.keys()), help="Forecast horizon")
    parser.add_argument("--CustomerCode", help="Customer code for sales prediction")
    parser.add_argument("--ItemSelectionType", choices=["single", "all"], help="Predict a single item or all items")

    args = parser.parse_args()
    engine = get_db_engine()
    prompt_file = Path(__file__).parent / "prompts" / "query_prompt.txt"
    columns = get_columns(args.prediction_type, prompt_file)
    query = build_query(columns, args.prediction_type, args.CustomerCode)
    df = pd.read_sql(query, engine)

    if df.empty:
        print("No data returned from query.")
        return

    series = df.iloc[:, -1].astype(float)
    series.index = pd.to_datetime(df.iloc[:, 0])
    steps = TIME_MAP[args.time_period]
    model = ForecastModel()
    forecast_mean, forecast_se = model.forecast(series, steps)
    output_dir = Path(__file__).parent / "outputs"
    plot_path = plot_forecast(series, forecast_mean, forecast_se, args.time_period, args.prediction_type, output_dir)
    print(f"Forecast saved to {plot_path}")


if __name__ == "__main__":
    main()
