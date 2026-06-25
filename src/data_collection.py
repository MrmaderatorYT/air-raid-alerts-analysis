import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional
import os
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


OBLAST_TO_CITY = {
    "Donetska oblast": "Donetsk",
    "Sumska oblast": "Sumy",
    "Zakarpatska oblast": "Uzhhorod",
    "Chernihivska oblast": "Chernihiv",
    "Kyivska oblast": "Kyiv",
    "Lvivska oblast": "Lviv",
    "Dnipropetrovska oblast": "Dnipro",
}


def fetch_alerts_from_api(
    start_date: str = "2022-02-24",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    all_alerts = []
    current_date = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while current_date <= end:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"https://api.ukrainealarm.com/api/v3/alerts/history/{date_str}"

        try:
            resp = requests.get(url, headers={"accept": "application/json"}, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for alert in data.get("alerts", []):
                    all_alerts.append({
                        "oblast": alert.get("oblast", ""),
                        "started_at": alert.get("startedAt"),
                        "finished_at": alert.get("finishedAt"),
                        "alert_type": alert.get("type", ""),
                    })
        except Exception as e:
            print(f"Error fetching {date_str}: {e}")

        current_date += timedelta(days=1)

    if not all_alerts:
        return pd.DataFrame()

    df = pd.DataFrame(all_alerts)
    return df


def fetch_alerts_batch(start_date: str = "2022-02-24", end_date: Optional[str] = None) -> pd.DataFrame:
    cache_file = DATA_DIR / "raw_api_cache.csv"

    if cache_file.exists():
        print(f"Loading cached data from {cache_file}")
        return pd.read_csv(cache_file)

    print("Fetching data from API...")
    df = fetch_alerts_from_api(start_date, end_date)

    if not df.empty:
        df.to_csv(cache_file, index=False)
        print(f"Data saved to {cache_file}")

    return df


def load_raw_csv(path: Optional[str] = None) -> pd.DataFrame:
    if path is None:
        path = DATA_DIR / "raw.csv"

    df = pd.read_csv(path, low_memory=False)

    required_cols = ["oblast", "started_at", "finished_at"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df


def get_city_data(
    df: pd.DataFrame,
    cities: Optional[list] = None,
) -> pd.DataFrame:
    if cities is None:
        cities = list(OBLAST_TO_CITY.values())

    df = df.copy()
    df["city"] = df["oblast"].map(OBLAST_TO_CITY)

    df = df[df["city"].isin(cities)].copy()

    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    kyiv_tz = "Europe/Kyiv"
    df["started_at"] = df["started_at"].dt.tz_convert(kyiv_tz)
    df["finished_at"] = df["finished_at"].dt.tz_convert(kyiv_tz)

    df["duration"] = df["finished_at"] - df["started_at"]
    df["duration_min"] = df["duration"].dt.total_seconds() / 60

    df = df[df["duration_min"] > 0].copy()
    df = df[df["duration_min"] < 24 * 60].copy()

    return df.reset_index(drop=True)


if __name__ == "__main__":
    csv_path = DATA_DIR / "raw.csv"

    if csv_path.exists():
        print("Loading from CSV...")
        df = load_raw_csv(csv_path)
    else:
        print("Fetching from API...")
        df = fetch_alerts_batch()

    if not df.empty:
        cities_df = get_city_data(df)
        print(f"City data shape: {cities_df.shape}")
        print(f"Cities: {cities_df['city'].unique()}")
        print(cities_df.head())
    else:
        print("No data available")
