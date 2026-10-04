########################################################################################################################
# IMPORTS
# You absolutely need these
import json
import mlflow
import mlflow.sklearn
import os

# You will probably need these
import pandas as pd
import numpy as np
import matplotlib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import skops.io as sio

matplotlib.use("Agg")

# This are for example purposes. You may discard them if you don't use them.
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

########################################################################################################################

## Step 1: The Data (from CSVs)
def read_csv_with_time_index(path):
    """Helper to read CSVs with a datetime index."""
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def create_eda_plots(joined_dfs):
    """
    Create exploratory data analysis plots for wind power data
    """
    fig, ax = plt.subplots(1, 3, figsize=(25, 4))

    # Speed and Power for the last 7 days
    ax[0].plot(joined_dfs["Speed"].tail(int(7 * 24 / 3)), label="Speed", color="blue")
    ax[0].plot(joined_dfs["Total"].tail(int(7 * 24 / 3)), label="Power", color="tab:red")
    ax[0].set_title("Windspeed & Power Generation over last 7 days")
    ax[0].set_xlabel("Time")
    ax[0].tick_params(axis="x", labelrotation=45)
    ax[0].set_ylabel("Windspeed [m/s], Power [MW]")
    ax[0].legend()

    # Speed vs Total (Power Curve nature)
    ax[1].scatter(joined_dfs["Speed"], joined_dfs["Total"])
    power_curve = joined_dfs.groupby("Speed").median(numeric_only=True)["Total"]
    ax[1].plot(power_curve.index, power_curve.values, "k:", label="Power Curve")
    ax[1].legend()
    ax[1].set_title("Windspeed vs Power")
    ax[1].set_ylabel("Power [MW]")
    ax[1].set_xlabel("Windspeed [m/s]")

    # Speed and Power per Wind Direction
    if "Direction" in joined_dfs.columns:
        wind_grouped_by_direction = joined_dfs.groupby("Direction").mean(numeric_only=True).reset_index()
        bar_width = 0.5
        x = np.arange(len(wind_grouped_by_direction.index))
        ax[2].bar(x, wind_grouped_by_direction.Total, width=0.5, label="Power", color="tab:red")
        ax[2].bar(x + bar_width, wind_grouped_by_direction.Speed, width=0.5, label="Speed", color="blue")
        ax[2].legend()
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(wind_grouped_by_direction.Direction)
        ax[2].tick_params(axis="x", labelrotation=45)
        ax[2].set_title("Speed and Power per Direction")
    else:
        ax[2].axis("off")

    plt.tight_layout()
    return fig

#Experiment pipeline and configuration

POWER_PATH = "data/power.csv"
WEATHER_PATH = "data/weather.csv"
FUTURE_PATH = "data/future.csv"
OUTPUT_DIR = "outputs"
TRACKING_URI = "sqlite:///mlflow.db"
REGISTERED_MODEL_NAME = "OrkneyWindPowerForecaster"
FINAL_SELECTION_EXPERIMENT = "WindPower-FinalSelection"
TEST_SIZE = 0.2
N_JOBS = 2

FEATURE_MODES = [
    "speed_only",
    "speed_direction",
    "wind_vector",
    "speed_and_wind_vector",
]


class WindFeatureTransformer(BaseEstimator, TransformerMixin):
    directions = [
        "N", "NNE", "NE", "ENE",
        "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW",
        "W", "WNW", "NW", "NNW",
    ]
    direction_to_degrees = {
        direction: index * 22.5
        for index, direction in enumerate(directions)
    }

    def __init__(self, mode="speed_direction"):
        self.mode = mode

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        speed = X["Speed"]

        if self.mode == "speed_only":
            return pd.DataFrame({"Speed": speed}, index=X.index)

        radians = np.deg2rad(
            X["Direction"].map(self.direction_to_degrees)
        )
        direction_sin = np.sin(radians)
        direction_cos = np.cos(radians)
        wind_x = speed * direction_cos
        wind_y = speed * direction_sin

        if self.mode == "speed_direction":
            return pd.DataFrame({
                "Speed": speed,
                "Direction_sin": direction_sin,
                "Direction_cos": direction_cos,
            }, index=X.index)

        if self.mode == "wind_vector":
            return pd.DataFrame({
                "Wind_x": wind_x,
                "Wind_y": wind_y,
            }, index=X.index)

        return pd.DataFrame({
            "Speed": speed,
            "Wind_x": wind_x,
            "Wind_y": wind_y,
        }, index=X.index)


def make_pipeline(model, feature_mode, polynomial=False, scale=False):
    steps = [("features", WindFeatureTransformer(mode=feature_mode))]

    if polynomial:
        steps.append(("poly", PolynomialFeatures(include_bias=False)))

    if scale:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))
    return Pipeline(steps)


def build_experiments():
    models = {
        "linear_regression": {
            "experiment_name": "WindPower-LinearRegression",
            "model": LinearRegression(),
            "polynomial": False,
            "scale": True,
            "grid": {},
        },
        "polynomial_ridge": {
            "experiment_name": "WindPower-PolynomialRidge",
            "model": Ridge(),
            "polynomial": True,
            "scale": True,
            "grid": {
                "poly__degree": [2, 3, 4],
                "model__alpha": [0.01, 0.1, 1, 10, 100],
            },
        },
        "random_forest": {
            "experiment_name": "WindPower-RandomForest",
            "model": RandomForestRegressor(random_state=42, n_jobs=1),
            "polynomial": False,
            "scale": False,
            "grid": {
                "model__n_estimators": [100, 300, 500],
                "model__max_depth": [5, 10, None],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": [0.7, 1.0],
            },
        },
        "svr_rbf": {
            "experiment_name": "WindPower-SVR-RBF",
            "model": SVR(kernel="rbf"),
            "polynomial": False,
            "scale": True,
            "grid": {
                "model__C": [0.1, 1, 10, 100, 300],
                "model__epsilon": [0.1, 0.5, 1.0],
                "model__gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
            },
        },
        "gradient_boosting": {
            "experiment_name": "WindPower-GradientBoosting",
            "model": GradientBoostingRegressor(random_state=42),
            "polynomial": False,
            "scale": False,
            "grid": {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [1, 2, 3],
                "model__min_samples_leaf": [3, 5, 10],
                "model__subsample": [0.8, 1.0],
            },
        },
    }

    experiments = {}
    for model_name, config in models.items():
        experiments[model_name] = {
            "experiment_name": config["experiment_name"],
            "pipeline": make_pipeline(
                clone(config["model"]),
                "speed_only",
                config["polynomial"],
                config["scale"],
            ),
            "grid": {
                "features__mode": FEATURE_MODES,
                **config["grid"],
            },
        }

    return experiments


def run_all_experiments():
    power_df = read_csv_with_time_index(POWER_PATH)
    wind_df = read_csv_with_time_index(WEATHER_PATH)
    
    power_df = power_df[power_df["Total"] >= 0]

    joined_dfs = power_df.resample("3h").mean().join(wind_df, how="inner")
    data = (
        joined_dfs[["Speed", "Direction", "Total"]]
        .sort_index()
        .dropna()
    )

    X = data[["Speed", "Direction"]]
    y = data["Total"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        shuffle=False,
    )

    experiments = build_experiments()
    time_cv = TimeSeriesSplit(n_splits=5,test_size=56)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    os.environ.pop("MLFLOW_RUN_ID", None)
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", TRACKING_URI))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    eda_fig = create_eda_plots(data)
    eda_fig.savefig("plots/eda_plots.png")
    plt.close(eda_fig)

    searches = {}
    results = []

    for number, (model_name, experiment) in enumerate(experiments.items(), start=1):
        print(f"[{number}/5] Running {model_name}")
        mlflow.set_experiment(experiment["experiment_name"])

        search = GridSearchCV(
            estimator=experiment["pipeline"],
            param_grid=experiment["grid"],
            scoring=scoring,
            refit="rmse",
            cv=time_cv,
            n_jobs=N_JOBS,
            return_train_score=True,
        )

        with mlflow.start_run(run_name=f"{model_name}_champion") as run:
            search.fit(X_train, y_train)
            i = search.best_index_
            best_model = search.best_estimator_
            train_predictions = best_model.predict(X_train)

            result = {
                "model": model_name,
                "feature_mode": search.best_params_["features__mode"],
                "training_mae": mean_absolute_error(y_train, train_predictions),
                "training_rmse": np.sqrt(mean_squared_error(y_train, train_predictions)),
                "training_r2": r2_score(y_train, train_predictions),
                "cv_train_mae": -search.cv_results_["mean_train_mae"][i],
                "cv_test_mae": -search.cv_results_["mean_test_mae"][i],
                "cv_test_mae_std": search.cv_results_["std_test_mae"][i],
                "cv_test_rmse": -search.cv_results_["mean_test_rmse"][i],
                "cv_test_r2": search.cv_results_["mean_test_r2"][i],
                "best_params": search.best_params_,
                "source_run_id": run.info.run_id,
            }

            searches[model_name] = search
            results.append(result)

            mlflow.log_params({
                "model": model_name,
                "pipeline": str(best_model),
                **{
                    f"best_{key}": str(value)
                    for key, value in search.best_params_.items()
                },
            })
            mlflow.log_metrics({
                key: result[key]
                for key in [
                    "training_mae",
                    "training_rmse",
                    "training_r2",
                    "cv_train_mae",
                    "cv_test_mae",
                    "cv_test_mae_std",
                    "cv_test_rmse",
                    "cv_test_r2",
                ]
            })

            cv_results = pd.DataFrame({
                "rank": search.cv_results_["rank_test_mae"],
                "parameters": [
                    json.dumps(params, default=str)
                    for params in search.cv_results_["params"]
                ],
                "train_mae": -search.cv_results_["mean_train_mae"],
                "cv_test_mae": -search.cv_results_["mean_test_mae"],
                "cv_test_mae_std": search.cv_results_["std_test_mae"],
                "cv_test_rmse": -search.cv_results_["mean_test_rmse"],
                "cv_test_r2": search.cv_results_["mean_test_r2"],
                "mean_fit_time": search.cv_results_["mean_fit_time"],
            }).sort_values("rank")

            mlflow.log_table(cv_results, "cv_results.json")
            mlflow.log_dict(experiment["grid"], "search_grid.json")

            candidate_signature = infer_signature(
                X_train.head(),
                best_model.predict(X_train.head()),
            )
            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="best_candidate_model",
                serialization_format="cloudpickle",
                signature=candidate_signature,
                input_example=X_train.head(),
            )

    comparison = pd.DataFrame([
        {
            **{key: value for key, value in result.items() if key != "best_params"},
            "best_params": json.dumps(result["best_params"], default=str),
        }
        for result in results
    ]).sort_values("cv_test_rmse").reset_index(drop=True)
    comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)

    best_name = comparison.loc[0, "model"]
    best_search = searches[best_name]
    best_model = best_search.best_estimator_

    mlflow.set_experiment(FINAL_SELECTION_EXPERIMENT)
    selection_run_ids = {}

    for result in results:
        model_name = result["model"]
        with mlflow.start_run(run_name=f"{model_name}_champion") as run:
            selection_run_ids[model_name] = run.info.run_id
            mlflow.log_params({
                "model": model_name,
                "source_run_id": result["source_run_id"],
                **{
                    f"best_{key}": str(value)
                    for key, value in result["best_params"].items()
                },
            })
            mlflow.log_metrics({
                key: result[key]
                for key in [
                    "training_mae",
                    "training_rmse",
                    "training_r2",
                    "cv_train_mae",
                    "cv_test_mae",
                    "cv_test_mae_std",
                    "cv_test_rmse",
                    "cv_test_r2",
                ]
            })
            mlflow.set_tag(
                "selection_status",
                "global_best" if model_name == best_name else "candidate",
            )

    test_predictions = best_model.predict(X_test)
    test_mae = mean_absolute_error(y_test, test_predictions)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    test_r2 = r2_score(y_test, test_predictions)

    pd.DataFrame({
        "observed_total": y_test,
        "predicted_total": test_predictions,
    }).to_csv(f"{OUTPUT_DIR}/test_predictions.csv", index_label="time")

    plt.figure(figsize=(15, 4))
    plt.plot(y_test.index, y_test, label="Truth")
    plt.plot(y_test.index, test_predictions, label="Predictions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/test_predictions.png")
    plt.close()

    final_model = clone(best_model).fit(X, y)
    future_df = read_csv_with_time_index(FUTURE_PATH)
    future_X = future_df[["Speed", "Direction"]]
    future_df["Predicted_Total"] = final_model.predict(future_X)
    future_df.to_csv(f"{OUTPUT_DIR}/future_predictions.csv", index_label="time")

    with mlflow.start_run(run_id=selection_run_ids[best_name]):
        mlflow.log_metrics({
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
        })
        mlflow.log_artifacts(OUTPUT_DIR)
        mlflow.log_artifact("plots/eda_plots.png")
        mlflow.log_artifact("plots/test_predictions.png")

        signature = infer_signature(
            X.head(),
            final_model.predict(X.head()),
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=X.head(),
            registered_model_name=REGISTERED_MODEL_NAME,
        )

    print("\nSelected model:", best_name)
    print("Best parameters:", best_search.best_params_)
    print(f"Test MAE:  {test_mae:.3f} MW")
    print(f"Test RMSE: {test_rmse:.3f} MW")
    print(f"Test R2:   {test_r2:.3f}")
    print("\nFirst 10 future predictions:")
    print(future_df[["Speed", "Direction", "Predicted_Total"]].head(10))
    print("\nRegistered model:", REGISTERED_MODEL_NAME)
    print("Model URI:", model_info.model_uri)


if __name__ == "__main__":
    run_all_experiments()
