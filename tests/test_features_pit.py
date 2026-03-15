import pandas as pd

from forecasting.features import build_pandas_features, spark_dayofweek


def test_lag_features_use_only_past_data():
    ds = pd.date_range("2024-01-01", periods=45, freq="D")
    base = pd.DataFrame(
        {
            "ds": ds,
            "store_id": 1,
            "sku_id": 101,
            "y": range(1, 46),
            "promo_flag": [0] * 45,
            "price": [10.0] * 45,
        }
    )
    feats = build_pandas_features(base)

    target_ds = pd.Timestamp("2024-02-10")
    row = feats.loc[feats["ds"] == target_ds].iloc[0]
    expected_lag7 = base.loc[base["ds"] == target_ds - pd.Timedelta(days=7), "y"].iloc[0]
    expected_lag14 = base.loc[base["ds"] == target_ds - pd.Timedelta(days=14), "y"].iloc[0]
    expected_lag28 = base.loc[base["ds"] == target_ds - pd.Timedelta(days=28), "y"].iloc[0]

    assert row["lag_7"] == expected_lag7
    assert row["lag_14"] == expected_lag14
    assert row["lag_28"] == expected_lag28


def test_rolling_mean_excludes_current_target():
    ds = pd.date_range("2024-01-01", periods=45, freq="D")
    base = pd.DataFrame(
        {
            "ds": ds,
            "store_id": 1,
            "sku_id": 101,
            "y": range(1, 46),
            "promo_flag": [0] * 45,
            "price": [10.0] * 45,
        }
    )
    feats = build_pandas_features(base)
    target_ds = pd.Timestamp("2024-02-10")
    row = feats.loc[feats["ds"] == target_ds].iloc[0]

    start = target_ds - pd.Timedelta(days=28)
    end = target_ds - pd.Timedelta(days=1)
    expected = base.loc[(base["ds"] >= start) & (base["ds"] <= end), "y"].mean()

    assert row["rolling_mean_28"] == expected
    assert row["rolling_mean_28"] != base.loc[base["ds"] == target_ds, "y"].iloc[0]


def test_spark_dayofweek_mapping_matches_spark_semantics():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-06", "2024-01-07"]))
    # Spark semantics: Monday=2, Saturday=7, Sunday=1
    mapped = spark_dayofweek(dates).tolist()
    assert mapped == [2, 7, 1]
