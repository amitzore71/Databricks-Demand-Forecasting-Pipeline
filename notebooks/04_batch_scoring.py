# Databricks notebook source
from __future__ import annotations

from datetime import datetime
import os
import pathlib
import sys

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient


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
    FEATURE_COLUMNS,
    FORECAST_HORIZON_DAYS,
    GOLD_DEMAND_FORECAST_TABLE,
    MODEL_NAME,
    SILVER_DEMAND_HISTORY_TABLE,
)
from forecasting.scoring import (  # noqa: E402
    build_future_exog,
    dedupe_forecasts,
    recursive_batch_forecast,
)


os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

try:
    champion = client.get_model_version_by_alias(MODEL_NAME, "Champion")
except Exception as exc:
    raise ValueError(
        "No Champion alias found. Run 03_register_champion.py and ensure threshold pass."
    ) from exc

model_uri = f"models:/{MODEL_NAME}@Champion"
model = mlflow.sklearn.load_model(model_uri=model_uri)
champion_version = str(champion.version)
champion_run_id = champion.run_id

feature_columns = FEATURE_COLUMNS
try:
    feature_meta = mlflow.artifacts.load_dict(f"runs:/{champion_run_id}/feature_columns.json")
    if isinstance(feature_meta, dict) and "feature_columns" in feature_meta:
        feature_columns = list(feature_meta["feature_columns"])
except Exception:
    pass

history_pdf = spark.table(SILVER_DEMAND_HISTORY_TABLE).select(
    "ds", "store_id", "sku_id", "y", "promo_flag", "price"
).toPandas()
if history_pdf.empty:
    raise ValueError("No history data available for batch scoring.")

future_exog_pdf = build_future_exog(history_pdf, horizon_days=FORECAST_HORIZON_DAYS)
pred_pdf = recursive_batch_forecast(
    history_df=history_pdf,
    future_exog_df=future_exog_pdf,
    model=model,
    feature_columns=feature_columns,
)
if pred_pdf.empty:
    raise ValueError("Recursive scoring produced no rows.")

pred_pdf["model_version"] = champion_version
pred_pdf["scored_ts"] = datetime.utcnow()
pred_pdf["ds"] = pd.to_datetime(pred_pdf["ds"]).dt.date
pred_pdf = dedupe_forecasts(
    pred_pdf,
    key_cols=["ds", "store_id", "sku_id", "model_version"],
)

scored_sdf = spark.createDataFrame(pred_pdf)
scored_sdf.createOrReplaceTempView("tmp_batch_forecast_scores")

spark.sql(
    f"""
    MERGE INTO {GOLD_DEMAND_FORECAST_TABLE} AS t
    USING tmp_batch_forecast_scores AS s
    ON t.ds = s.ds
       AND t.store_id = s.store_id
       AND t.sku_id = s.sku_id
       AND t.model_version = s.model_version
    WHEN MATCHED THEN UPDATE SET
      t.yhat = s.yhat,
      t.scored_ts = s.scored_ts
    WHEN NOT MATCHED THEN INSERT (
      ds, store_id, sku_id, yhat, model_version, scored_ts
    ) VALUES (
      s.ds, s.store_id, s.sku_id, s.yhat, s.model_version, s.scored_ts
    )
    """
)

print(
    f"Batch scoring complete: {len(pred_pdf)} rows for horizon {FORECAST_HORIZON_DAYS} days."
)
print(f"Champion model version: {champion_version}")
print(f"Upserted to: {GOLD_DEMAND_FORECAST_TABLE}")
