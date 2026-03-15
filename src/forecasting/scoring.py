"""Batch scoring helpers including recursive multi-step forecasting."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from forecasting.config import FEATURE_COLUMNS


def build_future_exog(history_df: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    """
    Build simple future exogenous features (promo_flag, price) for each series.

    Strategy:
    - For each future date, reuse values from the same day-of-week one week prior when present.
    - Fallback to the latest available values in the series.
    """
    required = {"ds", "store_id", "sku_id", "promo_flag", "price"}
    if not required.issubset(history_df.columns):
        raise ValueError(f"history_df missing required columns: {sorted(required)}")

    work = history_df.copy()
    work["ds"] = pd.to_datetime(work["ds"])
    work = work.sort_values(["store_id", "sku_id", "ds"]).reset_index(drop=True)
    global_last_date = work["ds"].max()

    future_rows: List[Dict] = []
    for (store_id, sku_id), g in work.groupby(["store_id", "sku_id"], sort=False):
        g = g.sort_values("ds")
        series_lookup = g.set_index("ds")[["promo_flag", "price"]].to_dict("index")
        last_promo = int(g["promo_flag"].iloc[-1])
        last_price = float(g["price"].iloc[-1])

        for step in range(1, horizon_days + 1):
            ds = global_last_date + timedelta(days=step)
            weekly_ref = ds - timedelta(days=7)
            if weekly_ref in series_lookup:
                promo_flag = int(series_lookup[weekly_ref]["promo_flag"])
                price = float(series_lookup[weekly_ref]["price"])
            else:
                promo_flag = last_promo
                price = last_price

            future_rows.append(
                {
                    "ds": ds,
                    "store_id": int(store_id),
                    "sku_id": int(sku_id),
                    "promo_flag": promo_flag,
                    "price": price,
                }
            )

    return pd.DataFrame(future_rows)


def _lag_value(y_by_date: Dict[pd.Timestamp, float], ds: pd.Timestamp, days: int) -> float:
    return float(y_by_date.get(ds - timedelta(days=days), np.nan))


def _rolling_mean_28(y_by_date: Dict[pd.Timestamp, float], ds: pd.Timestamp) -> float:
    vals = [y_by_date.get(ds - timedelta(days=i), np.nan) for i in range(1, 29)]
    vals = [v for v in vals if pd.notna(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _spark_dayofweek(ts: pd.Timestamp) -> int:
    """
    Spark-compatible day-of-week.

    Spark: Sunday=1 ... Saturday=7
    pandas: Monday=0 ... Sunday=6
    """
    return int(((ts.dayofweek + 1) % 7) + 1)


def recursive_batch_forecast(
    history_df: pd.DataFrame,
    future_exog_df: pd.DataFrame,
    model,
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Recursively predict horizon dates for each (store_id, sku_id) series.
    """
    if history_df.empty or future_exog_df.empty:
        return pd.DataFrame(columns=["ds", "store_id", "sku_id", "yhat"])

    hist = history_df.copy()
    fut = future_exog_df.copy()
    hist["ds"] = pd.to_datetime(hist["ds"])
    fut["ds"] = pd.to_datetime(fut["ds"])

    out_rows: List[Dict] = []
    for (store_id, sku_id), future_group in fut.groupby(["store_id", "sku_id"], sort=False):
        h = hist[(hist["store_id"] == store_id) & (hist["sku_id"] == sku_id)].sort_values("ds")
        y_by_date: Dict[pd.Timestamp, float] = {
            pd.Timestamp(d): float(y) for d, y in zip(h["ds"], h["y"])
        }
        if not y_by_date:
            continue

        fallback = float(h["y"].tail(28).mean())
        future_group = future_group.sort_values("ds")
        for _, row in future_group.iterrows():
            ds = pd.Timestamp(row["ds"])
            feature_row = {
                "promo_flag": float(row["promo_flag"]),
                "price": float(row["price"]),
                "dow": _spark_dayofweek(ds),
                "week_of_year": int(ds.isocalendar().week),
                "lag_7": _lag_value(y_by_date, ds, 7),
                "lag_14": _lag_value(y_by_date, ds, 14),
                "lag_28": _lag_value(y_by_date, ds, 28),
                "rolling_mean_28": _rolling_mean_28(y_by_date, ds),
            }

            # Guard for sparse historical series.
            for lag_col in ("lag_7", "lag_14", "lag_28", "rolling_mean_28"):
                if pd.isna(feature_row[lag_col]):
                    feature_row[lag_col] = fallback

            x_row = pd.DataFrame([feature_row], columns=list(feature_columns))
            yhat = float(model.predict(x_row)[0])
            yhat = max(yhat, 0.0)
            y_by_date[ds] = yhat
            out_rows.append(
                {
                    "ds": ds,
                    "store_id": int(store_id),
                    "sku_id": int(sku_id),
                    "yhat": yhat,
                }
            )

    return pd.DataFrame(out_rows)


def dedupe_forecasts(df: pd.DataFrame, key_cols: Iterable[str]) -> pd.DataFrame:
    """
    Keep latest row per idempotency key.

    If `scored_ts` exists, newest scored_ts wins.
    """
    keys = list(key_cols)
    if not keys:
        raise ValueError("key_cols cannot be empty")
    work = df.copy()
    if "scored_ts" in work.columns:
        work["scored_ts"] = pd.to_datetime(work["scored_ts"])
        work = work.sort_values("scored_ts")
    return work.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
