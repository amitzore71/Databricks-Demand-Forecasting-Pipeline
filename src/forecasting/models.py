"""Modeling and champion policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from forecasting.config import MIN_IMPROVEMENT_VS_NAIVE, TOP_N_SKUS_FOR_STABILITY


def safe_mape(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """MAPE that ignores zero targets to avoid divide-by-zero instability."""
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    mask = np.abs(y_true_arr) > 1e-9
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])))


def mean_error(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Mean signed error (bias proxy)."""
    y_true_arr = np.asarray(list(y_true), dtype=float)
    y_pred_arr = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(y_pred_arr - y_true_arr))


def compute_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Tuple[float, float]:
    """Return (mape, bias)."""
    return safe_mape(y_true, y_pred), mean_error(y_true, y_pred)


def top20_sku_mape_std(
    validation_df: pd.DataFrame, top_n: int = TOP_N_SKUS_FOR_STABILITY
) -> float:
    """
    Compute standard deviation of per-SKU MAPE for top-N SKUs by total demand.

    Expected columns:
    sku_id, y, yhat
    """
    if validation_df.empty:
        return float("nan")

    required_cols = {"sku_id", "y", "yhat"}
    if not required_cols.issubset(validation_df.columns):
        raise ValueError(f"Validation frame is missing required columns: {required_cols}")

    totals = validation_df.groupby("sku_id", as_index=False)["y"].sum()
    top_skus = totals.nlargest(top_n, "y")["sku_id"]
    top_df = validation_df[validation_df["sku_id"].isin(top_skus)].copy()

    mape_rows = []
    for sku_id, sku_df in top_df.groupby("sku_id", sort=False):
        mape_rows.append({"sku_id": sku_id, "mape": safe_mape(sku_df["y"], sku_df["yhat"])})

    per_sku_mape = pd.DataFrame(mape_rows)
    if per_sku_mape.empty or "mape" not in per_sku_mape.columns:
        return float("nan")
    return float(per_sku_mape["mape"].std(ddof=0))


@dataclass
class ChampionDecision:
    selected: Optional[Dict]
    reason: str


def choose_champion(
    run_rows: List[Dict], min_improvement: float = MIN_IMPROVEMENT_VS_NAIVE
) -> ChampionDecision:
    """
    Select champion based on policy:
    1) Must pass minimum improvement threshold vs naive baseline.
    2) Primary sort by lower validation MAPE.
    3) Tiebreakers: lower absolute bias, then lower top-20 SKU MAPE std.
    """
    model_rows = [r for r in run_rows if bool(r.get("has_model", True))]
    if not model_rows:
        return ChampionDecision(selected=None, reason="No model runs with registered artifacts.")

    eligible = [
        r
        for r in model_rows
        if float(r.get("improvement_vs_naive", float("-inf"))) >= min_improvement
    ]
    if not eligible:
        return ChampionDecision(
            selected=None,
            reason=(
                f"No run met minimum improvement threshold of "
                f"{min_improvement:.2%} vs naive baseline."
            ),
        )

    def sort_key(row: Dict) -> Tuple[float, float, float]:
        return (
            float(row["val_mape"]),
            abs(float(row["val_bias"])),
            float(row["top20_sku_mape_std"]),
        )

    winner = sorted(eligible, key=sort_key)[0]
    return ChampionDecision(selected=winner, reason="Winner selected by policy ordering.")
