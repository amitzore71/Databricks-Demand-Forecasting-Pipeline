"""Monitoring helpers for forecast quality and drift."""

from __future__ import annotations

from typing import Dict

import pandas as pd

from forecasting.models import compute_metrics


def calculate_segment_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Return dict with mape and bias from frame containing y and yhat."""
    if df.empty:
        return {"mape": float("nan"), "bias": float("nan")}
    mape, bias = compute_metrics(df["y"], df["yhat"])
    return {"mape": float(mape), "bias": float(bias)}


def make_drift_flag(
    current_mape: float,
    baseline_mape: float,
    current_bias: float,
    baseline_bias: float,
    mape_relative_threshold: float,
    bias_abs_shift_threshold: float,
) -> int:
    """Compute binary drift flag with MAPE and bias-shift triggers."""
    mape_trigger = current_mape > baseline_mape * (1.0 + mape_relative_threshold)
    bias_trigger = abs(current_bias - baseline_bias) > bias_abs_shift_threshold
    return int(mape_trigger or bias_trigger)

