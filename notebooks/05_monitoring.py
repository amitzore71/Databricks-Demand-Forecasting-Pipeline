# Databricks notebook source
from __future__ import annotations

import pathlib
import sys

import mlflow
from mlflow.tracking import MlflowClient
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
    DRIFT_BIAS_ABS_SHIFT_THRESHOLD,
    DRIFT_MAPE_RELATIVE_THRESHOLD,
    GOLD_DEMAND_FORECAST_TABLE,
    GOLD_FORECAST_MONITORING_TABLE,
    MODEL_NAME,
    ML_RUN_SUMMARY_TABLE,
    SILVER_DEMAND_HISTORY_TABLE,
)


mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

forecast_df = spark.table(GOLD_DEMAND_FORECAST_TABLE).select(
    "ds", "store_id", "sku_id", "yhat", "model_version"
)
actual_df = spark.table(SILVER_DEMAND_HISTORY_TABLE).select(
    "ds", "store_id", "sku_id", "y", "promo_flag"
)

joined_df = forecast_df.join(
    actual_df,
    on=["ds", "store_id", "sku_id"],
    how="inner",
).cache()

if joined_df.take(1) == []:
    print("No overlapping actuals with forecasts yet. Monitoring run skipped.")
else:
    top20_skus = (
        actual_df.where(F.col("ds") >= F.date_sub(F.current_date(), 90))
        .groupBy("sku_id")
        .agg(F.sum("y").alias("total_y"))
        .orderBy(F.desc("total_y"))
        .limit(20)
        .select("sku_id")
    )

    all_seg = joined_df.withColumn("segment", F.lit("ALL"))
    top_seg = (
        joined_df.join(top20_skus, on="sku_id", how="inner")
        .withColumn("segment", F.lit("TOP20_SKU"))
        .select(*joined_df.columns, "segment")
    )
    promo_seg = joined_df.where(F.col("promo_flag") == 1).withColumn("segment", F.lit("PROMO"))
    nonpromo_seg = joined_df.where(F.col("promo_flag") == 0).withColumn(
        "segment", F.lit("NON_PROMO")
    )

    stacked = all_seg.unionByName(top_seg).unionByName(promo_seg).unionByName(nonpromo_seg)
    metrics_df = (
        stacked.withColumn("error", F.col("yhat") - F.col("y"))
        .withColumn(
            "ape",
            F.when(F.abs(F.col("y")) > F.lit(1e-9), F.abs(F.col("error")) / F.abs(F.col("y"))),
        )
        .groupBy("ds", "segment")
        .agg(F.avg("ape").alias("mape"), F.avg("error").alias("bias"))
    )

    baseline_mape = 0.30
    baseline_bias = 0.0
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "Champion")
        run_id = champion.run_id
        baseline_row = (
            spark.table(ML_RUN_SUMMARY_TABLE)
            .where(F.col("run_id") == run_id)
            .orderBy(F.col("run_ts").desc())
            .first()
        )
        if baseline_row is not None:
            baseline_mape = float(baseline_row["val_mape"])
            baseline_bias = float(baseline_row["val_bias"])
    except Exception:
        pass

    monitored_df = (
        metrics_df.withColumn(
            "drift_flag",
            F.when(
                (F.col("mape") > F.lit(baseline_mape * (1.0 + DRIFT_MAPE_RELATIVE_THRESHOLD)))
                | (
                    F.abs(F.col("bias") - F.lit(baseline_bias))
                    > F.lit(DRIFT_BIAS_ABS_SHIFT_THRESHOLD)
                ),
                F.lit(1),
            ).otherwise(F.lit(0)),
        )
        .withColumn("evaluated_ts", F.current_timestamp())
        .select("ds", "segment", "mape", "bias", "drift_flag", "evaluated_ts")
    )

    monitored_df.createOrReplaceTempView("tmp_forecast_monitoring")
    spark.sql(
        f"""
        MERGE INTO {GOLD_FORECAST_MONITORING_TABLE} AS t
        USING tmp_forecast_monitoring AS s
        ON t.ds = s.ds AND t.segment = s.segment
        WHEN MATCHED THEN UPDATE SET
          t.mape = s.mape,
          t.bias = s.bias,
          t.drift_flag = s.drift_flag,
          t.evaluated_ts = s.evaluated_ts
        WHEN NOT MATCHED THEN INSERT (
          ds, segment, mape, bias, drift_flag, evaluated_ts
        ) VALUES (
          s.ds, s.segment, s.mape, s.bias, s.drift_flag, s.evaluated_ts
        )
        """
    )

    print(f"Monitoring updated in {GOLD_FORECAST_MONITORING_TABLE}")
    print(
        f"Baseline val_mape={baseline_mape:.4f}, baseline bias={baseline_bias:.4f}, "
        f"thresholds=({DRIFT_MAPE_RELATIVE_THRESHOLD:.0%} MAPE rel, {DRIFT_BIAS_ABS_SHIFT_THRESHOLD} bias abs)"
    )
