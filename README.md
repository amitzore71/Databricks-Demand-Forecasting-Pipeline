# Demand Forecasting Pipeline

This project predicts demand on Databricks. It manages the full machine learning lifecycle. It relies on MLflow for tracking and Delta Lake for storage.

## How it works

The code is modular. Each file in `src/forecasting` runs one specific stage.

### Feature Engineering

`features.py` calculates lags and rolling averages. It uses window functions for point-in-time correctness. This stops data leakage. The features capture seasonality and recent trends.

### Model Management

`models.py` defines the champion policy. It compares new models to a naive baseline. The script calculates MAPE and bias. It picks winners based on two primary filters:

- Models must beat the baseline by 5%.
- The winner has the lowest validation MAPE.

### Scoring and Monitoring

`scoring.py` runs batch predictions. Idempotent logic prevents duplicate rows in Delta tables. `monitoring.py` tracks performance. It raises flags when accuracy drops or bias shifts.

## Project Structure

- `src/forecasting/config.py`: Thresholds and table settings.
- `src/forecasting/features.py`: Data transformation logic.
- `src/forecasting/models.py`: Metrics and selection policy.
- `src/forecasting/scoring.py`: Batch inference.
- `src/forecasting/monitoring.py`: Drift detection.
- `notebooks/`: Orchestration notebooks.

## Setup

1. Install dependencies with `pip install -e .`.
2. Configure tables in `config.py`.
3. Run the training notebook.
4. Schedule the scoring job.
