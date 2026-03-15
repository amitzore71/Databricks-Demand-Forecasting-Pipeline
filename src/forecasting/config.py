"""Project constants shared by notebooks and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


CATALOG = "main"
SCHEMA = "retail_p3"
FULL_SCHEMA = f"{CATALOG}.{SCHEMA}"

SILVER_DEMAND_HISTORY_TABLE = f"{FULL_SCHEMA}.silver_demand_history"
ML_DEMAND_FEATURES_TABLE = f"{FULL_SCHEMA}.ml_demand_features"
ML_SPLIT_METADATA_TABLE = f"{FULL_SCHEMA}.ml_split_metadata"
ML_RUN_SUMMARY_TABLE = f"{FULL_SCHEMA}.ml_run_summary"
ML_CHAMPION_AUDIT_TABLE = f"{FULL_SCHEMA}.ml_champion_audit"
GOLD_DEMAND_FORECAST_TABLE = f"{FULL_SCHEMA}.gold_demand_forecast"
GOLD_FORECAST_MONITORING_TABLE = f"{FULL_SCHEMA}.gold_forecast_monitoring"

MODEL_NAME = f"{FULL_SCHEMA}.demand_forecast_model"
MLFLOW_EXPERIMENT_PATH = "/Shared/retail_p3_demand_forecasting"

FEATURE_COLUMNS: List[str] = [
    "promo_flag",
    "price",
    "dow",
    "week_of_year",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_28",
]

TRAIN_VALIDATION_RATIO = 0.8
FORECAST_HORIZON_DAYS = 14
MIN_IMPROVEMENT_VS_NAIVE = 0.15
TOP_N_SKUS_FOR_STABILITY = 20

DRIFT_MAPE_RELATIVE_THRESHOLD = 0.20
DRIFT_BIAS_ABS_SHIFT_THRESHOLD = 1.5


@dataclass(frozen=True)
class SplitConfig:
    """Global time split configuration."""

    split_name: str = "global_time_80_20"
    train_ratio: float = TRAIN_VALIDATION_RATIO

