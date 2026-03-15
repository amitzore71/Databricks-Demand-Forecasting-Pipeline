# Databricks notebook source
from __future__ import annotations

from datetime import datetime
import os
import pathlib
import sys

import mlflow
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
    MIN_IMPROVEMENT_VS_NAIVE,
    MODEL_NAME,
    ML_CHAMPION_AUDIT_TABLE,
    ML_RUN_SUMMARY_TABLE,
)
from forecasting.models import ChampionDecision, choose_champion  # noqa: E402


os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

run_summary_pdf = spark.table(ML_RUN_SUMMARY_TABLE).toPandas()
if run_summary_pdf.empty:
    raise ValueError("Run summary table is empty. Run 02_train_models.py first.")

run_records = run_summary_pdf.to_dict(orient="records")
decision: ChampionDecision = choose_champion(
    run_records, min_improvement=MIN_IMPROVEMENT_VS_NAIVE
)

selected_run_id = None
selected_version = None
threshold_passed = decision.selected is not None
reason = decision.reason


def _register_and_alias(model_uri: str, alias: str) -> str:
    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, alias, model_version.version)
    return str(model_version.version)


if decision.selected is not None:
    winner = decision.selected
    selected_run_id = str(winner["run_id"])
    selected_version = _register_and_alias(str(winner["model_uri"]), "Champion")
    client.set_registered_model_alias(MODEL_NAME, "Candidate", selected_version)
    reason = (
        f"{reason} Champion updated from run {selected_run_id} "
        f"with model version {selected_version}."
    )
else:
    candidate_df = run_summary_pdf[run_summary_pdf["has_model"] == True].copy()  # noqa: E712
    if not candidate_df.empty:
        candidate_df["abs_bias"] = candidate_df["val_bias"].abs()
        candidate_df = candidate_df.sort_values(
            by=["val_mape", "abs_bias", "top20_sku_mape_std"], ascending=True
        )
        candidate = candidate_df.iloc[0].to_dict()
        selected_run_id = str(candidate["run_id"])
        selected_version = _register_and_alias(str(candidate["model_uri"]), "Candidate")
        reason = (
            f"{reason} Candidate alias refreshed from run {selected_run_id} "
            f"with model version {selected_version}. Champion alias unchanged."
        )
    else:
        reason = f"{reason} No model artifact available to refresh Candidate alias."

audit_pdf = pd.DataFrame(
    [
        {
            "decision_ts": datetime.utcnow(),
            "selected_run_id": selected_run_id,
            "selected_version": selected_version,
            "threshold_passed": bool(threshold_passed),
            "reason": reason,
        }
    ]
)
spark.createDataFrame(audit_pdf).write.format("delta").mode("append").saveAsTable(
    ML_CHAMPION_AUDIT_TABLE
)

print("Champion selection completed.")
print(
    {
        "selected_run_id": selected_run_id,
        "selected_version": selected_version,
        "threshold_passed": threshold_passed,
        "reason": reason,
    }
)
