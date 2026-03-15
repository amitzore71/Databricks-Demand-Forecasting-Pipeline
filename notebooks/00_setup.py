# Databricks notebook source
from __future__ import annotations

import math
import pathlib
import sys

import mlflow
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
    FULL_SCHEMA,
    GOLD_DEMAND_FORECAST_TABLE,
    GOLD_FORECAST_MONITORING_TABLE,
    ML_CHAMPION_AUDIT_TABLE,
    ML_DEMAND_FEATURES_TABLE,
    ML_RUN_SUMMARY_TABLE,
    ML_SPLIT_METADATA_TABLE,
    MLFLOW_EXPERIMENT_PATH,
    SILVER_DEMAND_HISTORY_TABLE,
)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {FULL_SCHEMA}")

dates_df = spark.sql(
    """
    SELECT explode(
      sequence(to_date('2023-01-01'), to_date('2025-12-31'), interval 1 day)
    ) AS ds
    """
)
stores_df = spark.range(1, 11).selectExpr("CAST(id AS INT) AS store_id")
skus_df = spark.range(1, 51).selectExpr("CAST(id AS INT) AS sku_id")

base_df = dates_df.crossJoin(stores_df).crossJoin(skus_df)
base_df = (
    base_df.withColumn("dow", F.dayofweek("ds"))
    .withColumn("doy", F.dayofyear("ds"))
    .withColumn("promo_flag", (F.rand(seed=11) < F.lit(0.15)).cast("int"))
    .withColumn("sku_effect", (F.col("sku_id") % 10) * F.lit(1.8) + F.lit(18.0))
    .withColumn("store_effect", (F.col("store_id") % 5) * F.lit(1.2) + F.lit(4.0))
    .withColumn("base_price", F.lit(8.0) + (F.col("sku_id") % 15) * F.lit(1.5))
    .withColumn(
        "price",
        F.round(
            F.col("base_price")
            * F.when(F.col("promo_flag") == 1, F.lit(0.90)).otherwise(F.lit(1.0))
            * (F.lit(1.0) + (F.rand(seed=23) - F.lit(0.5)) * F.lit(0.03)),
            2,
        ),
    )
)

annual_seasonality = (
    F.sin((F.lit(2.0) * F.lit(math.pi) * F.col("doy")) / F.lit(365.0)) * F.lit(2.5)
)
weekly_seasonality = F.when(F.col("dow").isin([1, 7]), F.lit(3.5)).otherwise(F.lit(0.0))
promo_lift = F.col("promo_flag") * F.lit(7.0)
price_penalty = F.col("price") * F.lit(-0.55)
noise = F.randn(seed=37) * F.lit(2.2)

history_df = base_df.withColumn(
    "y",
    F.greatest(
        F.lit(0.0),
        F.col("sku_effect")
        + F.col("store_effect")
        + weekly_seasonality
        + annual_seasonality
        + promo_lift
        + price_penalty
        + noise,
    ),
).select(
    "ds",
    "store_id",
    "sku_id",
    F.round("y", 2).alias("y"),
    "promo_flag",
    "price",
)

history_df.write.format("delta").mode("overwrite").saveAsTable(SILVER_DEMAND_HISTORY_TABLE)

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {ML_DEMAND_FEATURES_TABLE} (
      ds DATE,
      store_id INT,
      sku_id INT,
      y DOUBLE,
      promo_flag INT,
      price DOUBLE,
      dow INT,
      week_of_year INT,
      lag_7 DOUBLE,
      lag_14 DOUBLE,
      lag_28 DOUBLE,
      rolling_mean_28 DOUBLE
    ) USING DELTA
    """
)

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {ML_SPLIT_METADATA_TABLE} (
      split_name STRING,
      cutoff_ds DATE,
      created_ts TIMESTAMP
    ) USING DELTA
    """
)

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {ML_RUN_SUMMARY_TABLE} (
      run_id STRING,
      model_type STRING,
      val_mape DOUBLE,
      val_bias DOUBLE,
      top20_sku_mape_std DOUBLE,
      improvement_vs_naive DOUBLE,
      has_model BOOLEAN,
      model_uri STRING,
      run_ts TIMESTAMP
    ) USING DELTA
    """
)

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {ML_CHAMPION_AUDIT_TABLE} (
      decision_ts TIMESTAMP,
      selected_run_id STRING,
      selected_version STRING,
      threshold_passed BOOLEAN,
      reason STRING
    ) USING DELTA
    """
)

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {GOLD_DEMAND_FORECAST_TABLE} (
      ds DATE,
      store_id INT,
      sku_id INT,
      yhat DOUBLE,
      model_version STRING,
      scored_ts TIMESTAMP
    ) USING DELTA
    """
)

spark.sql(
    f"""
    CREATE TABLE IF NOT EXISTS {GOLD_FORECAST_MONITORING_TABLE} (
      ds DATE,
      segment STRING,
      mape DOUBLE,
      bias DOUBLE,
      drift_flag INT,
      evaluated_ts TIMESTAMP
    ) USING DELTA
    """
)

mlflow.set_registry_uri("databricks-uc")
experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

print(f"Setup complete for schema: {FULL_SCHEMA}")
print(f"Seeded table: {SILVER_DEMAND_HISTORY_TABLE} -> {history_df.count()} rows")
print(f"MLflow experiment ID: {experiment.experiment_id}")

