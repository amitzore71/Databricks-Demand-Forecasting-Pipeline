# Databricks notebook source
from __future__ import annotations

from datetime import datetime
import pathlib
import sys

from pyspark.sql import Row
from pyspark.sql import Window
from pyspark.sql import functions as F


def _bootstrap_project_imports() -> None:
    cwd = pathlib.Path.cwd()
    for path in [cwd, *cwd.parents]:
        src_path = path / "src"
        if src_path.exists():
            sys.path.append(str(src_path))
            return
    raise RuntimeError("Could not find project src/ directory from current working directory.")


_bootstrap_project_imports()

from forecasting.config import (  # noqa: E402
    ML_DEMAND_FEATURES_TABLE,
    ML_SPLIT_METADATA_TABLE,
    SILVER_DEMAND_HISTORY_TABLE,
    SplitConfig,
)


history_df = spark.table(SILVER_DEMAND_HISTORY_TABLE).select(
    "ds", "store_id", "sku_id", "y", "promo_flag", "price"
)

w = Window.partitionBy("store_id", "sku_id").orderBy("ds")

features_df = (
    history_df.withColumn("dow", F.dayofweek("ds"))
    .withColumn("week_of_year", F.weekofyear("ds"))
    .withColumn("lag_7", F.lag("y", 7).over(w))
    .withColumn("lag_14", F.lag("y", 14).over(w))
    .withColumn("lag_28", F.lag("y", 28).over(w))
    .withColumn("rolling_mean_28", F.avg("y").over(w.rowsBetween(-28, -1)))
    .select(
        "ds",
        "store_id",
        "sku_id",
        "y",
        "promo_flag",
        "price",
        "dow",
        "week_of_year",
        "lag_7",
        "lag_14",
        "lag_28",
        "rolling_mean_28",
    )
)

features_df.write.format("delta").mode("overwrite").saveAsTable(ML_DEMAND_FEATURES_TABLE)

leakage_check_df = features_df.withColumn("lag_7_check", F.lag("y", 7).over(w))
violations = leakage_check_df.where(
    F.col("lag_7").isNotNull() & (F.abs(F.col("lag_7") - F.col("lag_7_check")) > F.lit(1e-8))
).count()
if violations != 0:
    raise ValueError(f"Point-in-time leakage check failed for lag_7 with {violations} rows.")

split_cfg = SplitConfig()
valid_dates_df = (
    features_df.where(F.col("lag_28").isNotNull() & F.col("rolling_mean_28").isNotNull())
    .select("ds")
    .distinct()
    .orderBy("ds")
)
date_count = valid_dates_df.count()
if date_count == 0:
    raise ValueError("No valid rows available after lag/rolling feature generation.")

cutoff_idx = max(0, int(date_count * split_cfg.train_ratio) - 1)
cutoff_ds = valid_dates_df.collect()[cutoff_idx]["ds"]

split_df = spark.createDataFrame(
    [
        Row(
            split_name=split_cfg.split_name,
            cutoff_ds=cutoff_ds,
            created_ts=datetime.utcnow(),
        )
    ]
)
split_df.write.format("delta").mode("overwrite").saveAsTable(ML_SPLIT_METADATA_TABLE)

print(f"Feature table written: {ML_DEMAND_FEATURES_TABLE}")
print(f"Split metadata written: {ML_SPLIT_METADATA_TABLE}")
print(f"Cutoff date: {cutoff_ds}")
print(f"Total feature rows: {features_df.count()}")

