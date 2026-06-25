import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


def train_arima(
    train: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
    seasonal_order: Optional[Tuple[int, int, int, int]] = None,
) -> Dict:
    try:
        if seasonal_order:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
        else:
            model = ARIMA(train, order=order)
            fitted = model.fit()

        return {
            "model": fitted,
            "aic": fitted.aic,
            "bic": fitted.bic,
            "order": order,
            "seasonal_order": seasonal_order,
        }
    except Exception as e:
        print(f"ARIMA training failed: {e}")
        return None


def train_prophet(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "total_duration_min",
    periods: int = 30,
) -> Dict:
    try:
        from prophet import Prophet

        prophet_df = df[[date_col, value_col]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        return {
            "model": model,
            "forecast": forecast,
            "components": model.plot_components(forecast),
        }
    except Exception as e:
        print(f"Prophet training failed: {e}")
        return None


def train_holt_winters(
    train: pd.Series,
    seasonal_periods: int = 7,
    trend: str = "add",
    seasonal: str = "add",
) -> Dict:
    try:
        model = ExponentialSmoothing(
            train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fitted = model.fit(optimized=True)

        return {
            "model": fitted,
            "aic": fitted.aic,
            "bic": fitted.bic,
        }
    except Exception as e:
        print(f"Holt-Winters training failed: {e}")
        return None


def evaluate_forecast(
    actual: pd.Series,
    predicted: pd.Series,
) -> Dict:
    min_len = min(len(actual), len(predicted))
    actual = actual.iloc[:min_len]
    predicted = predicted.iloc[:min_len]

    mask = actual > 0
    if mask.sum() == 0:
        return {"mae": 0, "rmse": 0, "mape": 0, "r2": 0}

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    actual_masked = actual[mask]
    predicted_masked = predicted[mask]
    mape = mean_absolute_percentage_error(actual_masked, predicted_masked) * 100

    ss_res = ((actual - predicted) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


def compare_models(
    train: pd.Series,
    test: pd.Series,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    arima_result = train_arima(train, order=(1, 1, 1))
    if arima_result:
        arima_pred = arima_result["model"].forecast(steps=len(test))
        arima_pred.index = test.index
        arima_metrics = evaluate_forecast(test, arima_pred)
        arima_metrics["model"] = "ARIMA(1,1,1)"
        arima_metrics["aic"] = arima_result["aic"]
        results.append(arima_metrics)

    hw_result = train_holt_winters(train)
    if hw_result:
        hw_pred = hw_result["model"].forecast(steps=len(test))
        hw_pred.index = test.index
        hw_metrics = evaluate_forecast(test, hw_pred)
        hw_metrics["model"] = "Holt-Winters"
        hw_metrics["aic"] = hw_result["aic"]
        results.append(hw_metrics)

    prophet_result = train_prophet(train_df)
    if prophet_result:
        prophet_forecast = prophet_result["forecast"]
        prophet_pred = prophet_forecast.tail(len(test))["yhat"].values
        prophet_pred = pd.Series(prophet_pred, index=test.index)
        prophet_metrics = evaluate_forecast(test, prophet_pred)
        prophet_metrics["model"] = "Prophet"
        prophet_metrics["aic"] = np.nan
        results.append(prophet_metrics)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def forecast_future(
    series: pd.Series,
    model_type: str = "arima",
    steps: int = 30,
    order: Tuple = (1, 1, 1),
) -> pd.Series:
    if model_type == "arima":
        result = train_arima(series, order=order)
        if result:
            forecast = result["model"].forecast(steps=steps)
            return forecast

    elif model_type == "holt_winters":
        result = train_holt_winters(series)
        if result:
            forecast = result["model"].forecast(steps=steps)
            return forecast

    return pd.Series([series.mean()] * steps)


if __name__ == "__main__":
    from data_collection import load_raw_csv, get_city_data
    from data_cleaning import remove_anomalies, create_daily_series, get_train_test_split

    csv_path = "../data/raw.csv"
    df = load_raw_csv(csv_path)
    cities_df = get_city_data(df)
    cleaned = remove_anomalies(cities_df)
    daily = create_daily_series(cleaned)

    city = "Kyiv"
    city_data = daily[daily["city"] == city].sort_values("date")

    train, test = get_train_test_split(city_data, test_days=30)
    train_series = train.set_index("date")["total_duration_min"]
    test_series = test.set_index("date")["total_duration_min"]

    comparison = compare_models(train_series, test_series, train, test)
    print("\nModel Comparison:")
    print(comparison[["model", "mae", "rmse", "mape", "r2"]].to_string(index=False))
