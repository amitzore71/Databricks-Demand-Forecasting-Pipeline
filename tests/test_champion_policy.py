from forecasting.models import choose_champion


def test_policy_rejects_models_below_minimum_improvement():
    rows = [
        {
            "run_id": "r1",
            "val_mape": 0.20,
            "val_bias": 0.1,
            "top20_sku_mape_std": 0.05,
            "improvement_vs_naive": 0.10,
            "has_model": True,
        },
        {
            "run_id": "r2",
            "val_mape": 0.19,
            "val_bias": -0.2,
            "top20_sku_mape_std": 0.04,
            "improvement_vs_naive": 0.12,
            "has_model": True,
        },
    ]

    decision = choose_champion(rows, min_improvement=0.15)
    assert decision.selected is None
    assert "minimum improvement" in decision.reason


def test_policy_tiebreaks_on_bias_then_stability():
    rows = [
        {
            "run_id": "r_best_mape_high_bias",
            "val_mape": 0.16,
            "val_bias": 0.30,
            "top20_sku_mape_std": 0.03,
            "improvement_vs_naive": 0.20,
            "has_model": True,
        },
        {
            "run_id": "r_same_mape_better_bias",
            "val_mape": 0.16,
            "val_bias": 0.05,
            "top20_sku_mape_std": 0.07,
            "improvement_vs_naive": 0.20,
            "has_model": True,
        },
        {
            "run_id": "r_same_mape_same_bias_better_stability",
            "val_mape": 0.16,
            "val_bias": -0.05,
            "top20_sku_mape_std": 0.02,
            "improvement_vs_naive": 0.20,
            "has_model": True,
        },
    ]

    decision = choose_champion(rows, min_improvement=0.15)
    assert decision.selected is not None
    assert decision.selected["run_id"] == "r_same_mape_same_bias_better_stability"

