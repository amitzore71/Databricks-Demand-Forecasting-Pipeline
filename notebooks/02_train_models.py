# Databricks notebook source
from __future__ import annotations

from datetime import datetime
import os
import pathlib
import sys
from typing import Dict, List

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


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
    MIN_IMPROVEMENT_VS_NAIVE,
    MLFLOW_EXPERIMENT_PATH,
    ML_DEMAND_FEATURES_TABLE,
    ML_RUN_SUMMARY_TABLE,
    ML_SPLIT_METADATA_TABLE,
)
from forecasting.models import compute_metrics, top20_sku_mape_std  # noqa: E402


os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)

split_meta_row = spark.table(ML_SPLIT_METADATA_TABLE).orderBy("created_ts", ascending=False).first()
if split_meta_row is None:
    raise ValueError("Missing split metadata. Run 01_feature_pipeline.py first.")
cutoff_ds = pd.Timestamp(split_meta_row["cutoff_ds"])

df = spark.table(ML_DEMAND_FEATURES_TABLE).where(
    "lag_28 IS NOT NULL AND rolling_mean_28 IS NOT NULL"
).toPandas()
if df.empty:
    raise ValueError("No rows in feature table after null filtering.")

df["ds"] = pd.to_datetime(df["ds"])
df = df.sort_values(["ds", "store_id", "sku_id"]).reset_index(drop=True)

train_df = df[df["ds"] <= cutoff_ds].copy()
val_df = df[df["ds"] > cutoff_ds].copy()
if train_df.empty or val_df.empty:
    raise ValueError("Train/validation split produced an empty partition.")

X_tr = train_df[FEATURE_COLUMNS]
y_tr = train_df["y"]
X_va = val_df[FEATURE_COLUMNS]
y_va = val_df["y"]

run_summaries: List[Dict] = []

# Baseline run: yhat = lag_7
naive_pred = val_df["lag_7"].values
naive_mape, naive_bias = compute_metrics(y_va, naive_pred)
naive_top20_std = top20_sku_mape_std(val_df.assign(yhat=naive_pred))

with mlflow.start_run(run_name="naive_lag7_baseline") as run:
    mlflow.log_param("model_type", "NaiveLag7")
    mlflow.log_param("cutoff_ds", cutoff_ds.strftime("%Y-%m-%d"))
    mlflow.log_metric("val_mape", float(naive_mape))
    mlflow.log_metric("val_bias", float(naive_bias))
    mlflow.log_metric("top20_sku_mape_std", float(naive_top20_std))
    mlflow.log_metric("improvement_vs_naive", 0.0)
    mlflow.log_dict({"feature_columns": FEATURE_COLUMNS}, "feature_columns.json")

    run_summaries.append(
        {
            "run_id": run.info.run_id,
            "model_type": "NaiveLag7",
            "val_mape": float(naive_mape),
            "val_bias": float(naive_bias),
            "top20_sku_mape_std": float(naive_top20_std),
            "improvement_vs_naive": 0.0,
            "has_model": False,
            "model_uri": None,
            "run_ts": datetime.utcnow(),
        }
    )


model_specs = [
    ("linear_regression", LinearRegression()),
    ("random_forest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ("hist_gbdt", HistGradientBoostingRegressor(random_state=42, max_depth=8)),
]

for run_name, model in model_specs:
    with mlflow.start_run(run_name=run_name) as run:
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        mape, bias = compute_metrics(y_va, pred)
        top20_std = top20_sku_mape_std(val_df.assign(yhat=pred))
        improvement = (
            (naive_mape - mape) / naive_mape if pd.notna(naive_mape) and abs(naive_mape) > 1e-12 else 0.0
        )

        mlflow.log_param("model_type", type(model).__name__)
        mlflow.log_param("cutoff_ds", cutoff_ds.strftime("%Y-%m-%d"))
        mlflow.log_param("min_required_improvement", MIN_IMPROVEMENT_VS_NAIVE)
        mlflow.log_metric("val_mape", float(mape))
        mlflow.log_metric("val_bias", float(bias))
        mlflow.log_metric("top20_sku_mape_std", float(top20_std))
        mlflow.log_metric("improvement_vs_naive", float(improvement))
        mlflow.log_dict({"feature_columns": FEATURE_COLUMNS}, "feature_columns.json")
        input_example = X_tr.head(5)
        signature = infer_signature(input_example, model.predict(input_example))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )

        sample = val_df[["ds", "store_id", "sku_id", "y"]].copy()
        sample["yhat"] = pred
        sample["ds"] = sample["ds"].astype(str)
        mlflow.log_dict(
            {"rows": sample.head(200).to_dict(orient="records")},
            "validation/predictions_sample.json",
        )

        run_summaries.append(
            {
                "run_id": run.info.run_id,
                "model_type": type(model).__name__,
                "val_mape": float(mape),
                "val_bias": float(bias),
                "top20_sku_mape_std": float(top20_std),
                "improvement_vs_naive": float(improvement),
                "has_model": True,
                "model_uri": f"runs:/{run.info.run_id}/model",
                "run_ts": datetime.utcnow(),
            }
        )

summary_pdf = pd.DataFrame(run_summaries)
summary_sdf = spark.createDataFrame(summary_pdf)
summary_sdf.write.format("delta").mode("overwrite").saveAsTable(ML_RUN_SUMMARY_TABLE)

print(f"Training complete with {len(run_summaries)} runs.")
print(f"Run summary written to {ML_RUN_SUMMARY_TABLE}")
print(summary_pdf[["model_type", "val_mape", "val_bias", "improvement_vs_naive"]].sort_values("val_mape"))
