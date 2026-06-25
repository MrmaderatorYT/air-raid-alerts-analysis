import pandas as pd
import numpy as np
from typing import Tuple


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_anomaly"] = False
    df["anomaly_reason"] = ""

    duration_99 = df["duration_min"].quantile(0.99)
    long_mask = df["duration_min"] > duration_99
    df.loc[long_mask, "is_anomaly"] = True
    df.loc[long_mask, "anomaly_reason"] = "extreme_duration"

    for city in df["city"].unique():
        city_data = df[df["city"] == city]
        if len(city_data) < 10:
            continue

        q1 = city_data["duration_min"].quantile(0.25)
        q3 = city_data["duration_min"].quantile(0.75)
        iqr = q3 - q1

        city_mask = df["city"] == city
        outlier_mask = city_mask & (df["duration_min"] > q3 + 3 * iqr)
        df.loc[outlier_mask, "is_anomaly"] = True
        df.loc[outlier_mask, "anomaly_reason"] = "iqr_outlier"

    return df


def remove_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = detect_anomalies(df)

    anomaly_count = df["is_anomaly"].sum()
    print(f"Removed {anomaly_count} anomalies ({anomaly_count/len(df)*100:.2f}%)")

    df = df[~df["is_anomaly"]].copy()
    df = df.drop(columns=["is_anomaly", "anomaly_reason"])

    return df.reset_index(drop=True)


def create_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["started_at"].dt.date

    daily = df.groupby(["city", "date"]).agg(
        total_duration_min=("duration_min", "sum"),
        alert_count=("duration_min", "count"),
        avg_duration_min=("duration_min", "mean"),
        max_duration_min=("duration_min", "max"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])

    return daily


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["started_at"].dt.hour

    def time_bucket(hour):
        if 23 <= hour or hour < 6:
            return "night"
        elif 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        else:
            return "evening"

    df["time_bucket"] = df["hour"].apply(time_bucket)

    df["is_night"] = df["time_bucket"] == "night"

    df["year"] = df["started_at"].dt.year
    df["month"] = df["started_at"].dt.month
    df["day_of_week"] = df["started_at"].dt.dayofweek
    df["week_of_year"] = df["started_at"].dt.isocalendar().week.astype(int)

    return df


def create_night_share_series(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df["date"] = df["started_at"].dt.date

    daily = df.groupby(["city", "date"]).agg(
        total_duration=("duration_min", "sum"),
        night_duration=("duration_min", lambda x: x[df.loc[x.index, "is_night"]].sum()),
    ).reset_index()

    daily["night_share"] = daily["night_duration"] / daily["total_duration"]
    daily["night_share"] = daily["night_share"].fillna(0)

    daily["date"] = pd.to_datetime(daily["date"])

    return daily


def prepare_forecasting_data(
    daily: pd.DataFrame,
    city: str,
) -> pd.DataFrame:
    city_data = daily[daily["city"] == city].copy()
    city_data = city_data.sort_values("date")

    date_range = pd.date_range(
        start=city_data["date"].min(),
        end=city_data["date"].max(),
        freq="D",
    )

    city_data = city_data.set_index("date").reindex(date_range)
    city_data = city_data.fillna(0)
    city_data.index.name = "date"
    city_data = city_data.reset_index()

    return city_data


def get_train_test_split(
    df: pd.DataFrame,
    test_days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date")
    split_idx = len(df) - test_days

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    return train, test


if __name__ == "__main__":
    from data_collection import load_raw_csv, get_city_data

    csv_path = "../data/raw.csv"
    df = load_raw_csv(csv_path)
    cities_df = get_city_data(df)

    print("Original shape:", cities_df.shape)

    cleaned = remove_anomalies(cities_df)
    print("Cleaned shape:", cleaned.shape)

    daily = create_daily_series(cleaned)
    print("Daily series shape:", daily.shape)
    print("Sample:\n", daily.head(10))
