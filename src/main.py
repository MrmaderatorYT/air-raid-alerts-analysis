import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from data_collection import load_raw_csv, get_city_data, fetch_alerts_batch
from data_cleaning import (
    remove_anomalies,
    create_daily_series,
    create_night_share_series,
    get_train_test_split,
    prepare_forecasting_data,
)
from analysis import create_analysis_summary, find_optimal_params
from forecasting import compare_models, forecast_future
from visualization import (
    plot_daily_duration_by_city,
    plot_time_bucket_analysis,
    plot_night_share,
    plot_heatmap,
    plot_forecast_comparison,
)


DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


CITIES = ["Kyiv", "Kharkiv", "Lviv", "Dnipro", "Zaporizhzhia", "Uzhhorod", "Sumy", "Chernihiv"]

FOCUS_CITIES = ["Kyiv", "Kharkiv", "Lviv", "Dnipro", "Zaporizhzhia", "Uzhhorod", "Sumy", "Chernihiv"]


def load_data() -> pd.DataFrame:
    csv_path = DATA_DIR / "raw.csv"

    if csv_path.exists():
        print("Loading from CSV...")
        df = load_raw_csv(csv_path)
    else:
        print("Fetching from API...")
        df = fetch_alerts_batch()

    return df


def run_analysis_pipeline():
    print("=" * 60)
    print("UKRAINE AIR ALERT ANALYSIS")
    print("=" * 60)

    raw_df = load_data()
    if raw_df.empty:
        print("ERROR: No data available")
        return

    print(f"\nRaw data shape: {raw_df.shape}")

    cities_df = get_city_data(raw_df, cities=CITIES)
    print(f"City data shape: {cities_df.shape}")
    print(f"Cities: {cities_df['city'].unique()}")

    cleaned_df = remove_anomalies(cities_df)
    print(f"Cleaned data shape: {cleaned_df.shape}")

    daily_df = create_daily_series(cleaned_df)
    print(f"Daily series shape: {daily_df.shape}")

    night_df = create_night_share_series(cleaned_df)

    print("\n" + "=" * 60)
    print("EXPLORATORY ANALYSIS")
    print("=" * 60)

    plot_daily_duration_by_city(daily_df, cities=FOCUS_CITIES)
    plot_time_bucket_analysis(cleaned_df, cities=FOCUS_CITIES)
    plot_night_share(night_df, cities=FOCUS_CITIES)

    for city in FOCUS_CITIES:
        plot_heatmap(daily_df, city)

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    summaries = {}
    for city in CITIES:
        print(f"\n--- {city} ---")
        summary = create_analysis_summary(daily_df, city)
        summaries[city] = summary

        print(f"  Days: {summary['n_days']}")
        print(f"  Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
        print(f"  Mean duration: {summary['mean_duration']:.1f} min")
        print(f"  Max duration: {summary['max_duration']:.1f} min")
        print(f"  Alert ratio: {summary['alert_ratio']:.1%}")
        print(f"  Stationary: {summary['stationarity']['is_stationary']}")

    print("\n" + "=" * 60)
    print("TIME SERIES FORECASTING")
    print("=" * 60)

    forecast_results = {}
    for city in CITIES:
        print(f"\n--- {city} ---")

        city_data = daily_df[daily_df["city"] == city].sort_values("date")
        forecast_data = prepare_forecasting_data(city_data, city)

        train, test = get_train_test_split(forecast_data, test_days=30)
        train_series = train.set_index("date")["total_duration_min"]
        test_series = test.set_index("date")["total_duration_min"]

        comparison = compare_models(train_series, test_series, train, test)

        if not comparison.empty:
            print(comparison[["model", "mae", "rmse", "mape", "r2"]].to_string(index=False))

            best_model = comparison.loc[comparison["mape"].idxmin(), "model"]
            print(f"\nBest model: {best_model}")

            forecast_results[city] = {
                "comparison": comparison,
                "best_model": best_model,
                "train": train,
                "test": test,
            }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary_data = []
    for city, result in forecast_results.items():
        summary_data.append({
            "City": city,
            "Best Model": result["best_model"],
            "MAPE (%)": result["comparison"].loc[
                result["comparison"]["model"] == result["best_model"], "mape"
            ].values[0],
            "R²": result["comparison"].loc[
                result["comparison"]["model"] == result["best_model"], "r2"
            ].values[0],
        })

    summary_df = pd.DataFrame(summary_data)
    print("\nForecast Model Summary:")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(OUTPUT_DIR / "model_summary.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")

    return {
        "daily": daily_df,
        "night": night_df,
        "summaries": summaries,
        "forecasts": forecast_results,
    }


if __name__ == "__main__":
    results = run_analysis_pipeline()
