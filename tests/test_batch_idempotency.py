import pandas as pd

from forecasting.scoring import dedupe_forecasts


def test_dedupe_forecasts_keeps_latest_scored_ts():
    df = pd.DataFrame(
        [
            {
                "ds": "2025-01-01",
                "store_id": 1,
                "sku_id": 100,
                "model_version": "7",
                "yhat": 10.0,
                "scored_ts": "2026-01-01T00:00:00",
            },
            {
                "ds": "2025-01-01",
                "store_id": 1,
                "sku_id": 100,
                "model_version": "7",
                "yhat": 11.5,
                "scored_ts": "2026-01-01T01:00:00",
            },
            {
                "ds": "2025-01-01",
                "store_id": 2,
                "sku_id": 100,
                "model_version": "7",
                "yhat": 8.0,
                "scored_ts": "2026-01-01T00:30:00",
            },
        ]
    )
    df["ds"] = pd.to_datetime(df["ds"])

    out = dedupe_forecasts(df, key_cols=["ds", "store_id", "sku_id", "model_version"])
    assert len(out) == 2

    key = (
        (out["ds"] == pd.Timestamp("2025-01-01"))
        & (out["store_id"] == 1)
        & (out["sku_id"] == 100)
        & (out["model_version"] == "7")
    )
    selected = out.loc[key].iloc[0]
    assert selected["yhat"] == 11.5

