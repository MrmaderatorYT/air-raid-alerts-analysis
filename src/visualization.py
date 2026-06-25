import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Optional, List
from pathlib import Path


IMAGES_DIR = Path(__file__).parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

plt.rcParams["figure.figsize"] = (14, 5)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10
sns.set_theme(style="whitegrid")


def plot_daily_duration_by_city(
    daily_df: pd.DataFrame,
    cities: Optional[List[str]] = None,
    save: bool = True,
) -> None:
    if cities is None:
        cities = daily_df["city"].unique()

    fig, axes = plt.subplots(len(cities), 1, figsize=(14, 4 * len(cities)), sharex=True)
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        city_data = daily_df[daily_df["city"] == city].sort_values("date")
        ax.fill_between(city_data["date"], city_data["total_duration_min"], alpha=0.3)
        ax.plot(city_data["date"], city_data["total_duration_min"], linewidth=0.5)

        rolling = city_data["total_duration_min"].rolling(30, min_periods=1).mean()
        ax.plot(city_data["date"], rolling, color="red", linewidth=2, label="30-day MA")

        ax.set_title(city, fontweight="bold")
        ax.set_ylabel("Duration (min)")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    if save:
        fig.savefig(IMAGES_DIR / "daily_duration_by_city.png", bbox_inches="tight")
    plt.show()


def plot_time_bucket_analysis(
    df: pd.DataFrame,
    cities: Optional[List[str]] = None,
    save: bool = True,
) -> None:
    from data_cleaning import add_time_features

    df = add_time_features(df)

    if cities is None:
        cities = df["city"].unique()

    fig, axes = plt.subplots(len(cities), 1, figsize=(14, 4 * len(cities)), sharex=True)
    if len(cities) == 1:
        axes = [axes]

    bucket_colors = {"night": "#1a1a2e", "morning": "#e67e22", "afternoon": "#f39c12", "evening": "#8e44ad"}

    for ax, city in zip(axes, cities):
        city_data = df[df["city"] == city].copy()
        city_data["date"] = city_data["started_at"].dt.date

        daily_buckets = city_data.groupby(["date", "time_bucket"])["duration_min"].sum().unstack(fill_value=0)
        daily_buckets = daily_buckets.reindex(columns=["night", "morning", "afternoon", "evening"], fill_value=0)

        daily_buckets.index = pd.to_datetime(daily_buckets.index)
        daily_buckets.plot(kind="area", stacked=True, ax=ax, color=[bucket_colors[b] for b in daily_buckets.columns])

        ax.set_title(city, fontweight="bold")
        ax.set_ylabel("Duration (min)")
        ax.legend(title="Time", loc="upper right")

    axes[-1].set_xlabel("Date")
    plt.tight_layout()

    if save:
        fig.savefig(IMAGES_DIR / "time_bucket_analysis.png", bbox_inches="tight")
    plt.show()


def plot_night_share(
    daily_night_df: pd.DataFrame,
    cities: Optional[List[str]] = None,
    save: bool = True,
) -> None:
    if cities is None:
        cities = daily_night_df["city"].unique()

    fig, ax = plt.subplots(figsize=(14, 6))

    for city in cities:
        city_data = daily_night_df[daily_night_df["city"] == city].sort_values("date")
        rolling = city_data["night_share"].rolling(30, min_periods=7).mean()
        ax.plot(city_data["date"], rolling, label=city, linewidth=1.5)

    ax.set_title("Night Share of Alert Duration (30-day Rolling Average)", fontweight="bold")
    ax.set_ylabel("Night Share (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    plt.tight_layout()

    if save:
        fig.savefig(IMAGES_DIR / "night_share_analysis.png", bbox_inches="tight")
    plt.show()


def plot_weekly_pattern(
    weekly_df: pd.DataFrame,
    city: str,
    save: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(weekly_df.index, weekly_df["mean_duration"], color="#3498db", alpha=0.7)

    ax.errorbar(
        weekly_df.index,
        weekly_df["mean_duration"],
        yerr=weekly_df["std_duration"],
        fmt="none",
        color="red",
        capsize=3,
    )

    ax.set_title(f"Weekly Pattern - {city}", fontweight="bold")
    ax.set_ylabel("Mean Duration (min)")
    ax.set_xlabel("Day of Week")

    plt.tight_layout()

    if save:
        fig.savefig(IMAGES_DIR / f"weekly_pattern_{city.lower()}.png", bbox_inches="tight")
    plt.show()


def plot_yearly_trend(
    yearly_df: pd.DataFrame,
    city: str,
    save: bool = True,
) -> None:
    fig, ax1 = plt.subplots(figsize=(12, 5))

    color1 = "#3498db"
    color2 = "#e74c3c"

    ax1.bar(yearly_df.index, yearly_df["total_duration"], color=color1, alpha=0.7, label="Total Duration")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Duration (min)", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(yearly_df.index, yearly_df["mean_duration"], color=color2, marker="o", linewidth=2, label="Mean Daily Duration")
    ax2.set_ylabel("Mean Daily Duration (min)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.set_title(f"Yearly Trend - {city}", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()

    if save:
        fig.savefig(IMAGES_DIR / f"yearly_trend_{city.lower()}.png", bbox_inches="tight")
    plt.show()


def plot_forecast_comparison(
    train: pd.Series,
    test: pd.Series,
    forecasts: dict,
    city: str,
    save: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(train.index[-60:], train.values[-60:], label="Train (last 60 days)", color="gray", alpha=0.7)
    ax.plot(test.index, test.values, label="Actual", color="black", linewidth=2)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for (name, forecast), color in zip(forecasts.items(), colors):
        ax.plot(test.index, forecast.values, label=name, color=color, linewidth=1.5, linestyle="--")

    ax.set_title(f"Forecast Comparison - {city}", fontweight="bold")
    ax.set_ylabel("Duration (min)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")

    plt.tight_layout()

    if save:
        fig.savefig(IMAGES_DIR / f"forecast_comparison_{city.lower()}.png", bbox_inches="tight")
    plt.show()


def plot_heatmap(
    df: pd.DataFrame,
    city: str,
    save: bool = True,
) -> None:
    city_data = df[df["city"] == city].copy()
    city_data["date"] = pd.to_datetime(city_data["date"])
    city_data["year"] = city_data["date"].dt.year
    city_data["month"] = city_data["date"].dt.month

    pivot = city_data.pivot_table(
        values="total_duration_min",
        index="year",
        columns="month",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        ax=ax,
        cbar_kws={"label": "Mean Duration (min)"},
    )

    ax.set_title(f"Monthly Alert Duration Heatmap - {city}", fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    plt.tight_layout()

    if save:
        fig.savefig(IMAGES_DIR / f"heatmap_{city.lower()}.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    from data_collection import load_raw_csv, get_city_data
    from data_cleaning import remove_anomalies, create_daily_series, create_night_share_series

    csv_path = "../data/raw.csv"
    df = load_raw_csv(csv_path)
    cities_df = get_city_data(df)
    cleaned = remove_anomalies(cities_df)
    daily = create_daily_series(cleaned)

    plot_daily_duration_by_city(daily)
    plot_heatmap(daily, "Kyiv")
