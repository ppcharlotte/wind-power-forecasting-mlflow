# Wind Power Forecasting with MLflow

A reproducible machine learning pipeline for forecasting wind power production in Orkney from weather inputs. The project combines temporal data alignment, wind-feature engineering, time-series model selection, and MLflow experiment tracking and model registration.

Developed as a course project and adapted from [BDM-Assignment1](https://github.com/ppcharlotte/BDM-Assignment1).

## Approach

1. Remove negative power measurements, aggregate power to three-hour means, and join weather observations by timestamp.
2. Drop rows with missing modeling inputs or targets and reserve the final 20% of observations as a chronological test set.
3. Compare four feature representations: wind speed alone; speed with sine/cosine wind direction; wind-vector components; and speed with wind-vector components.
4. Tune five regression model families using a shared five-fold `TimeSeriesSplit`, with 56 observations per validation fold.
5. Select each model family's best configuration by cross-validation RMSE, then select the overall winner using the same criterion.
6. Evaluate the winner on the held-out test period, refit it on all historical data, and forecast the weather inputs in `future.csv`.
7. Log experiments and artifacts and register the final pipeline as `OrkneyWindPowerForecaster`.

Preprocessing and estimation are combined in scikit-learn pipelines so scaling and polynomial transformations are fitted within each training fold.

### Models and Evaluation

- Linear regression
- Polynomial regression with Ridge regularization
- Random forest
- RBF support vector regression
- Gradient boosting

Evaluation uses MAE, RMSE, and R². The grid-search result table ranks candidates by MAE, while model selection uses RMSE.

## Quick Start

Use Python 3.14 and run commands from the repository root.

```bash
git clone https://github.com/ppcharlotte/wind-power-forecasting-mlflow.git
cd wind-power-forecasting-mlflow

python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python script.py
```

The full hyperparameter search can take substantial time. Paths, model grids, feature modes, and validation settings are configured in `script.py`; the search uses two parallel jobs.

Alternatively, run the packaged MLflow Project:

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db mlflow run . \
  --experiment-name WindPower-ProjectRuns
```

The project environment is specified in `python_env.yaml`. MLflow environment management may require additional local environment tooling.

## Data

The repository includes:

| File | Purpose |
| --- | --- |
| `data/power.csv` | Historical power production; the target is `Total` |
| `data/weather.csv` | Historical weather inputs, including `Speed` and `Direction` |
| `data/future.csv` | Weather inputs for future predictions |

All CSV files use a `time` column. Wind directions are mapped from 16 compass points to angles, then encoded as circular features or speed-weighted vector components.

## Track Experiments

Start the local MLflow UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open [http://localhost:5001](http://localhost:5001).

Each model family has a separate experiment containing its best configuration, training and validation metrics, search grid, complete cross-validation table, and candidate model. `WindPower-FinalSelection` brings the five family winners together for comparison and records the final winner's test metrics and registered model.

The first training run creates a local `mlflow.db` and artifact directories. These generated files are excluded from version control. To use a separate tracking database:

```bash
MLFLOW_TRACKING_URI=sqlite:///fresh_mlflow.db python script.py
```

Use the same tracking URI when opening the UI or accessing the registered model.

## Outputs

| Path | Contents |
| --- | --- |
| `outputs/model_comparison.csv` | Cross-validation results for the five family winners |
| `outputs/test_predictions.csv` | Observed and predicted power for the held-out period |
| `outputs/future_predictions.csv` | Future weather inputs with `Predicted_Total` |
| `plots/eda_plots.png` | Exploratory wind and power plots |
| `plots/test_predictions.png` | Test-period predictions versus observations |
| MLflow registry | Final refitted pipeline, input signature, and example input |

The script also prints the selected model, test metrics, and first ten future predictions.

## Repository Structure

```text
├── README.md          # Project overview and usage
├── script.py          # Data preparation, training, evaluation, and registration
├── MLproject          # MLflow Project entry point
├── python_env.yaml    # MLflow-managed Python environment
├── requirements.txt   # Pinned dependencies
├── .gitignore         # Excludes local environments and experiment artifacts
└── data/
    ├── power.csv      # Historical power production
    ├── weather.csv    # Historical weather inputs
    └── future.csv     # Weather inputs for forecasting
```

## Scope and Limitations

This repository implements a local training and model-registration workflow. The training script does not start a prediction API; model serving is a separate step.

Missing training rows are dropped rather than imputed. A persistence baseline, additional historical-window comparisons, and deployment validation would strengthen the evaluation. Chronological splitting preserves temporal order, but operational use also requires checking forecast issue times and target-window alignment against the actual prediction horizon.

The full training search has not been rerun as part of preparing this public copy. Report quantitative performance from a reproducible run with its configuration and artifacts.
