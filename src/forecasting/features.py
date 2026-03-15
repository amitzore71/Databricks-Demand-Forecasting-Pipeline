"""Feature engineering helpers."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from forecasting.config import TRAIN_VALIDATION_RATIO


def spark_dayofweek(series: pd.Series) -> pd.Series:
    """
    Return Spark-compatible day-of-week values from pandas timestamps.

    Spark: Sunday=1, Monday=2, ..., Saturday=7
    pandas dt.dayofweek: Monday=0, ..., Sunday=6
    """
    return ((series.dt.dayofweek + 1) % 7) + 1


def build_pandas_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build point-in-time safe demand features using pandas.

    Expected input columns:
    ds, store_id, sku_id, y, promo_flag, price
    """
    required_cols = {"ds", "store_id", "sku_id", "y", "promo_flag", "price"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature generation: {sorted(missing)}")

    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"])
    out = out.sort_values(["store_id", "sku_id", "ds"]).reset_index(drop=True)

    out["dow"] = spark_dayofweek(out["ds"])
    out["week_of_year"] = out["ds"].dt.isocalendar().week.astype(int)

    grouped = out.groupby(["store_id", "sku_id"], sort=False)
    out["lag_7"] = grouped["y"].shift(7)
    out["lag_14"] = grouped["y"].shift(14)
    out["lag_28"] = grouped["y"].shift(28)
    # Rolling window only over prior days; current row excluded via shift(1).
    out["rolling_mean_28"] = grouped["y"].shift(1).rolling(28, min_periods=28).mean()

    return out


def compute_global_cutoff(ds_values: pd.Series, train_ratio: float = TRAIN_VALIDATION_RATIO) -> pd.Timestamp:
    """Compute the global cutoff date for deterministic time-based split."""
    if ds_values.empty:
        raise ValueError("Cannot compute cutoff for empty date series")

    unique_dates = sorted(pd.to_datetime(ds_values).unique())
    idx = max(0, int(len(unique_dates) * train_ratio) - 1)
    return pd.Timestamp(unique_dates[idx])


def split_train_validation(
    df: pd.DataFrame, cutoff_ds: pd.Timestamp
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by global cutoff date."""
    work = df.copy()
    work["ds"] = pd.to_datetime(work["ds"])
    train_df = work[work["ds"] <= cutoff_ds].copy()
    val_df = work[work["ds"] > cutoff_ds].copy()
    return train_df, val_df
