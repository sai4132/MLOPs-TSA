import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import subprocess
import yaml
from prefect import flow, task
import os
from config import AIR_PASSENGERS_CSV, AIR_PASSENGERS_DVC, REGISTRY_PATH

def load_registry(path=REGISTRY_PATH):
    if not os.path.exists(path):
        return {"production": {}}
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_registry(registry, path=REGISTRY_PATH):
    with open(path, "w") as f:
        yaml.safe_dump(registry, f)


# -----------------------
# Pipeline steps
# -----------------------
@task
def quality_check(rmse, threshold=100.0):
    passed = rmse < threshold
    print(f"Quality check | RMSE={rmse:.2f} | Passed={passed}")
    return passed

@task
def load_data(path):
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Month"])
    df["t"] = np.arange(len(df))
    return df

@task
def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"

@task
def get_dvc_data_hash(dvc_file_path):
    try:
        with open(dvc_file_path, "r") as f:
            dvc_data = yaml.safe_load(f)
        return dvc_data["outs"][0]["md5"]
    except Exception:
        return "unknown"

@task(retries=2, retry_delay_seconds=5)
def train_and_evaluate(df, forecast_window):
    train = df.iloc[:-forecast_window]
    test = df.iloc[-forecast_window:]

    X_train = train[["t"]]
    y_train = train["Passengers"]
    X_test = test[["t"]]
    y_test = test["Passengers"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    return model, rmse

@task
def log_experiment(model, rmse, metadata):
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("forecast_window", metadata["forecast_window"])
    mlflow.log_param("git_commit", metadata["git_commit"])
    mlflow.log_param("dvc_data_hash", metadata["dvc_data_hash"])

    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, artifact_path="model")

# -----------------------
# Pipeline orchestration
# -----------------------


def train_pipeline():
    df = load_data(AIR_PASSENGERS_CSV)

    git_commit = get_git_commit()
    data_hash = get_dvc_data_hash(AIR_PASSENGERS_DVC)

    for window in [12, 24, 36]:
        with mlflow.start_run():
            metadata = {
                "forecast_window": window,
                "git_commit": git_commit,
                "dvc_data_hash": data_hash
            }
            model, rmse = train_and_evaluate(df, window)
            log_experiment(model, rmse, metadata)

            passed = quality_check(rmse)

            if passed:
                registry = load_registry()
                prod_rmse = registry["production"].get("rmse")

                promote = (
                    prod_rmse is None or rmse < prod_rmse
                )

                if promote:
                    registry["production"] = {
                        "run_id": mlflow.active_run().info.run_id,
                        "model_path": "model",
                        "rmse": float(rmse),  # <-- FIX
                        "git_commit": git_commit,
                        "dvc_data_hash": data_hash,
                    }
                    save_registry(registry)
                    mlflow.log_param("model_status", "promoted")
                    print("Model promoted to production")
                else:
                    mlflow.log_param("model_status", "accepted_not_promoted")
                    print("Model accepted but not promoted")
            else:
                mlflow.log_param("model_status", "rejected")
                print("Model rejected")

            print(f"Window={window} | RMSE={rmse}")

@flow(name="tsa-training-pipeline")
def run_pipeline():
    train_pipeline()

if __name__ == "__main__":
    if os.getenv("DISABLE_PREFECT", "0") == "1":
        print("Running training WITHOUT Prefect orchestration")
        train_pipeline()
    else:
        print("Running training WITH Prefect orchestration")
        run_pipeline()