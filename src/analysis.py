import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Optional


def test_stationarity(series: pd.Series) -> Dict[str, float]:
    result = adfuller(series.dropna(), autolag="AIC")

    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
        "critical_values": result[4],
        "is_stationary": result[1] < 0.05,
    }


def decompose_series(
    series: pd.Series,
    period: int = 7,
    model: str = "additive",
) -> Dict[str, pd.Series]:
    series = series.fillna(0)

    decomp = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")

    return {
        "observed": decomp.observed,
        "trend": decomp.trend,
        "seasonal": decomp.seasonal,
        "residual": decomp.resid,
    }


def calculate_rolling_stats(
    series: pd.Series,
    windows: list = [7, 14, 30],
) -> pd.DataFrame:
    result = pd.DataFrame({"value": series})

    for w in windows:
        result[f"rolling_mean_{w}"] = series.rolling(window=w, min_periods=1).mean()
        result[f"rolling_std_{w}"] = series.rolling(window=w, min_periods=1).std()

    return result


def analyze_weekly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_name"] = pd.to_datetime(df["date"]).dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    weekly = df.groupby("day_name").agg(
        mean_duration=("total_duration_min", "mean"),
        std_duration=("total_duration_min", "std"),
        median_duration=("total_duration_min", "median"),
        mean_count=("alert_count", "mean"),
    ).reindex(day_order)

    return weekly


def analyze_monthly_pattern(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month_name"] = pd.to_datetime(df["date"]).dt.month_name()

    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    monthly = df.groupby("month_name").agg(
        mean_duration=("total_duration_min", "mean"),
        std_duration=("total_duration_min", "std"),
        mean_count=("alert_count", "mean"),
    ).reindex(month_order)

    return monthly


def analyze_yearly_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    yearly = df.groupby("year").agg(
        total_duration=("total_duration_min", "sum"),
        mean_duration=("total_duration_min", "mean"),
        total_alerts=("alert_count", "sum"),
        days_with_alerts=("total_duration_min", lambda x: (x > 0).sum()),
    )

    yearly["alerts_per_day"] = yearly["total_alerts"] / len(df["date"].unique())

    return yearly


def analyze_night_pattern(df: pd.DataFrame) -> pd.DataFrame:
    if "is_night" not in df.columns:
        from data_cleaning import add_time_features
        df = add_time_features(df)

    daily = df.groupby(["city", "date"]).agg(
        total_duration=("duration_min", "sum"),
        night_duration=("duration_min", lambda x: x[df.loc[x.index, "is_night"]].sum()),
    ).reset_index()

    daily["night_share"] = daily["night_duration"] / daily["total_duration"]
    daily["night_share"] = daily["night_share"].fillna(0)

    return daily


def find_optimal_params(series: pd.Series) -> Dict[str, int]:
    from pmdarima import auto_arima

    series = series.fillna(0)

    try:
        model = auto_arima(
            series,
            seasonal=True,
            m=7,
            suppress_warnings=True,
            error_action="ignore",
            stepwise=True,
        )

        return {
            "order": model.order,
            "seasonal_order": model.seasonal_order,
            "aic": model.aic(),
            "bic": model.bic(),
        }
    except Exception as e:
        print(f"Auto ARIMA failed: {e}")
        return {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 7), "aic": np.nan, "bic": np.nan}


def create_analysis_summary(
    daily_df: pd.DataFrame,
    city: str,
) -> Dict:
    city_data = daily_df[daily_df["city"] == city].copy()
    city_data = city_data.sort_values("date")

    series = city_data.set_index("date")["total_duration_min"]

    stationarity = test_stationarity(series)

    weekly_pattern = analyze_weekly_pattern(city_data)
    monthly_pattern = analyze_monthly_pattern(city_data)
    yearly_trend = analyze_yearly_trend(city_data)

    return {
        "city": city,
        "n_days": len(city_data),
        "date_range": (city_data["date"].min(), city_data["date"].max()),
        "mean_duration": city_data["total_duration_min"].mean(),
        "std_duration": city_data["total_duration_min"].std(),
        "max_duration": city_data["total_duration_min"].max(),
        "days_with_alerts": (city_data["total_duration_min"] > 0).sum(),
        "alert_ratio": (city_data["total_duration_min"] > 0).mean(),
        "stationarity": stationarity,
        "weekly_pattern": weekly_pattern,
        "monthly_pattern": monthly_pattern,
        "yearly_trend": yearly_trend,
    }


if __name__ == "__main__":
    from data_collection import load_raw_csv, get_city_data
    from data_cleaning import remove_anomalies, create_daily_series

    csv_path = "../data/raw.csv"
    df = load_raw_csv(csv_path)
    cities_df = get_city_data(df)
    cleaned = remove_anomalies(cities_df)
    daily = create_daily_series(cleaned)

    kyiv_summary = create_analysis_summary(daily, "Kyiv")
    print(f"Kyiv analysis:")
    print(f"  Days: {kyiv_summary['n_days']}")
    print(f"  Mean duration: {kyiv_summary['mean_duration']:.1f} min")
    print(f"  Stationary: {kyiv_summary['stationarity']['is_stationary']}")
