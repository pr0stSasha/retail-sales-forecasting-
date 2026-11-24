import os

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

HORIZONS = {
    "week": 7,
    "month": 30,
    "quarter": 90,
}

MODEL_TITLES = {
    "holt_winters": "Holt-Winters (ETS)",
    "arima": "ARIMA(1,1,1)",
    "sarima": "SARIMA(1,1,1)(0,1,1,7)",
}


class HoltWintersModel:
    name = "holt_winters"

    def __init__(self, week_len=7):
        self.week_len = week_len

    def fit(self, train_frame):
        tmp = train_frame.sort_values("date")
        series_vals = tmp["y"].to_numpy(dtype=float)
        ets_model = ExponentialSmoothing(
            series_vals,
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=self.week_len,
            initialization_method="estimated",
        )
        self.ets_fit = ets_model.fit(optimized=True)
        return self

    def predict(self, future_frame):
        preds = self.ets_fit.forecast(len(future_frame))
        return np.clip(preds, 0, None)


class ArimaModel:
    name = "arima"

    def __init__(self, arima_order=(1, 1, 1)):
        self.arima_order = arima_order

    def fit(self, train_frame):
        tmp = train_frame.sort_values("date")
        series_vals = tmp["y"].to_numpy(dtype=float)
        arima_model = ARIMA(series_vals, order=self.arima_order)
        self.arima_fit = arima_model.fit()
        return self

    def predict(self, future_frame):
        preds = self.arima_fit.forecast(len(future_frame))
        return np.clip(preds, 0, None)


class SarimaModel:
    name = "sarima"

    def fit(self, train_frame):
        tmp = train_frame.sort_values("date")
        series_vals = tmp["y"].to_numpy(dtype=float)
        sarima_model = SARIMAX(
            series_vals,
            order=(1, 1, 1),
            seasonal_order=(0, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.sarima_fit = sarima_model.fit(disp=False, maxiter=100)
        return self

    def predict(self, future_frame):
        preds = self.sarima_fit.forecast(len(future_frame))
        return np.clip(preds, 0, None)


MODELS = {
    "holt_winters": HoltWintersModel,
    "arima": ArimaModel,
    "sarima": SarimaModel,
}


class SalesForecaster:
    def __init__(self, store_id="STORE_1", data_dir="data", test_days=90):
        self.store_id = store_id
        self.data_dir = data_dir
        self.test_days = test_days
        self.sales = None
        self.calendar = None
        self.data = None
        self.train = None
        self.test = None
        self.test_start = None
        self.items = []
        self.models = {}
        self.forecasts = {}
        self.metrics = None

    def load_data(self):
        self.sales = pd.read_csv(
            os.path.join(self.data_dir, "shop_sales.csv")
        )
        self.calendar = pd.read_csv(
            os.path.join(self.data_dir, "shop_sales_dates.csv")
        )
        return self

    def preprocess(self):
        if self.sales is None or self.calendar is None:
            raise RuntimeError("Сначала вызовите load_data().")
        tmp_sales = self.sales[self.sales["store_id"] == self.store_id]
        tmp_sales = tmp_sales.copy()
        tmp_cal = self.calendar[["date_id", "date"]]
        tmp_df = tmp_sales.merge(tmp_cal, on="date_id", how="left")
        tmp_df["date"] = pd.to_datetime(tmp_df["date"])
        tmp_df = tmp_df.sort_values(["item_id", "date"])
        tmp_df = tmp_df.reset_index(drop=True)
        tmp_df = tmp_df.rename(columns={"cnt": "y"})
        self.data = tmp_df
        self.items = sorted(tmp_df["item_id"].unique())
        return self

    def split(self):
        if self.data is None:
            raise RuntimeError("Сначала вызовите preprocess().")
        all_dates = np.sort(self.data["date"].unique())
        n_days = len(all_dates)
        if self.test_days >= n_days:
            raise ValueError("Слишком большая тестовая выборка.")
        start_idx = n_days - self.test_days
        self.test_start = pd.Timestamp(all_dates[start_idx])
        train_mask = self.data["date"] < self.test_start
        test_mask = self.data["date"] >= self.test_start
        self.train = self.data[train_mask].copy()
        self.test = self.data[test_mask].copy()
        return self

    def fit(self, models=None):
        if self.train is None or self.test is None:
            raise RuntimeError("Сначала вызовите split().")
        if models is None:
            model_names = list(MODELS)
        else:
            model_names = list(models)
        bad_names = set(model_names) - set(MODELS)
        if bad_names:
            raise ValueError(f"Неизвестные модели: {sorted(bad_names)}")
        self.models = {model_key: {} for model_key in model_names}
        for i in range(len(self.items)):
            item_code = self.items[i]
            item_rows = self.train[self.train["item_id"] == item_code]
            item_train = item_rows.sort_values("date")
            for j in range(len(model_names)):
                model_key = model_names[j]
                fitted_model = MODELS[model_key]()
                fitted_model.fit(item_train)
                self.models[model_key][item_code] = fitted_model
        self.forecasts = {}
        return self

    def predict(self, model_name, horizon):
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name!r} не обучена.")
        if self.test is None:
            raise RuntimeError("Сначала вызовите split().")
        part_frames = []
        for i in range(len(self.items)):
            item_code = self.items[i]
            item_rows = self.test[self.test["item_id"] == item_code]
            item_test = item_rows.sort_values("date")
            future_rows = item_test.head(horizon)
            if len(future_rows) < horizon:
                raise ValueError(
                    "Горизонт больше доступной тестовой выборки."
                )
            model_obj = self.models[model_name][item_code]
            pred_values = model_obj.predict(future_rows)
            part_frame = future_rows[["item_id", "date", "y"]].copy()
            part_frame["y_pred"] = pred_values
            part_frame["model"] = model_name
            part_frames.append(part_frame)
        result_frame = pd.concat(part_frames, ignore_index=True)
        self.forecasts[(model_name, horizon)] = result_frame
        return result_frame

    def calculate_metrics(self, fact, prediction):
        f_arr = np.array(fact, dtype=float)
        p_arr = np.array(prediction, dtype=float)
        n_obs = len(f_arr)
        abs_err = []
        sq_err = []
        for i in range(n_obs):
            dlt = f_arr[i] - p_arr[i]
            abs_err.append(abs(dlt))
            sq_err.append(dlt * dlt)
        mae = sum(abs_err) / n_obs
        rmse = (sum(sq_err) / n_obs) ** 0.5
        sum_fact = sum(abs(x) for x in f_arr)
        if sum_fact == 0:
            wape = np.nan
        else:
            wape = sum(abs_err) / sum_fact
        smape_list = []
        for i in range(n_obs):
            den = abs(f_arr[i]) + abs(p_arr[i])
            if den > 0:
                smape_list.append(2 * abs_err[i] / den)
            else:
                smape_list.append(0)
        smape = sum(smape_list) / n_obs
        bias = sum(p_arr[i] - f_arr[i] for i in range(n_obs)) / n_obs
        return {
            "MAE": mae,
            "RMSE": rmse,
            "WAPE": wape,
            "sMAPE": smape,
            "bias": bias,
        }

    def horizon_name(self, horizon):
        if horizon == 7:
            return "week"
        elif horizon == 30:
            return "month"
        elif horizon == 90:
            return "quarter"
        else:
            return str(horizon) + "d"

    def evaluate(self, horizons=None):
        if not self.models:
            raise RuntimeError("Сначала вызовите fit().")
        if horizons is None:
            horizon_list = sorted(HORIZONS.values())
        else:
            horizon_list = list(horizons)
        metric_rows = []
        for model_key in self.models:
            for days in horizon_list:
                tmp_pred = self.predict(model_key, days)
                tmp_metrics = self.calculate_metrics(
                    tmp_pred["y"],
                    tmp_pred["y_pred"],
                )
                sum_fact = tmp_pred["y"].sum()
                sum_pred = tmp_pred["y_pred"].sum()
                if sum_fact > 0:
                    err_pct = (sum_pred - sum_fact) / sum_fact * 100
                else:
                    err_pct = np.nan
                metric_rows.append(
                    {
                        "model": model_key,
                        "model_title": MODEL_TITLES.get(
                            model_key,
                            model_key,
                        ),
                        "horizon_name": self.horizon_name(days),
                        "horizon": days,
                        **tmp_metrics,
                        "total_fact": sum_fact,
                        "total_pred": sum_pred,
                        "total_error_pct": err_pct,
                    }
                )
        col_order = [
            "model",
            "model_title",
            "horizon_name",
            "horizon",
            "MAE",
            "RMSE",
            "WAPE",
            "sMAPE",
            "bias",
            "total_fact",
            "total_pred",
            "total_error_pct",
        ]
        self.metrics = pd.DataFrame(metric_rows)[col_order]
        return self.metrics

    def aggregate_forecast(self, model_name, horizon):
        tmp_pred = self.predict(model_name, horizon)
        return tmp_pred.groupby("date", as_index=False).agg(
            fact=("y", "sum"),
            forecast=("y_pred", "sum"),
        )

    def evaluate_aggregated(self, horizons=None):
        if not self.models:
            raise RuntimeError("Сначала вызовите fit().")
        if horizons is None:
            horizon_list = sorted(HORIZONS.values())
        else:
            horizon_list = list(horizons)
        metric_rows = []
        for model_key in self.models:
            for days in horizon_list:
                tmp_daily = self.aggregate_forecast(model_key, days)
                tmp_metrics = self.calculate_metrics(
                    tmp_daily["fact"],
                    tmp_daily["forecast"],
                )
                sum_fact = tmp_daily["fact"].sum()
                sum_pred = tmp_daily["forecast"].sum()
                if sum_fact > 0:
                    err_pct = (sum_pred - sum_fact) / sum_fact * 100
                else:
                    err_pct = np.nan
                metric_rows.append(
                    {
                        "model": model_key,
                        "model_title": MODEL_TITLES.get(
                            model_key,
                            model_key,
                        ),
                        "horizon_name": self.horizon_name(days),
                        "horizon": days,
                        **tmp_metrics,
                        "total_fact": sum_fact,
                        "total_pred": sum_pred,
                        "total_error_pct": err_pct,
                    }
                )
        col_order = [
            "model",
            "model_title",
            "horizon_name",
            "horizon",
            "MAE",
            "RMSE",
            "WAPE",
            "sMAPE",
            "bias",
            "total_fact",
            "total_pred",
            "total_error_pct",
        ]
        return pd.DataFrame(metric_rows)[col_order]

    def forecast_totals(self, model_name, horizon):
        tmp_pred = self.predict(model_name, horizon)
        tmp_totals = (
            tmp_pred.groupby("item_id")
            .agg(fact=("y", "sum"), forecast=("y_pred", "sum"))
            .reset_index()
        )
        tmp_totals["error_pct"] = np.where(
            tmp_totals["fact"] > 0,
            (
                (tmp_totals["forecast"] - tmp_totals["fact"])
                / tmp_totals["fact"]
                * 100
            ),
            np.nan,
        )
        return tmp_totals

    def best_model(self, horizon, metric="WAPE"):
        if self.metrics is None:
            raise RuntimeError("Сначала вызовите evaluate().")
        tmp_metrics = self.metrics[self.metrics["horizon"] == horizon]
        best_idx = tmp_metrics[metric].idxmin()
        return str(tmp_metrics.loc[best_idx, "model"])

    def metrics_pivot(self, metric="WAPE"):
        if self.metrics is None:
            raise RuntimeError("Сначала вызовите evaluate().")
        return self.metrics.pivot_table(
            index="model_title",
            columns="horizon_name",
            values=metric,
        )
