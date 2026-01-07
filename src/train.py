import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import subprocess
import yaml

# -----------------------
# Pipeline steps
# -----------------------

def load_data(path):
    df = pd.read_csv(path)
    df["Month"] = pd.to_datetime(df["Month"])
    df["t"] = np.arange(len(df))
    return df

def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode("utf-8").strip()
    except Exception:
        return "unknown"

def get_dvc_data_hash(dvc_file_path):
    try:
        with open(dvc_file_path, "r") as f:
            dvc_data = yaml.safe_load(f)
        return dvc_data["outs"][0]["md5"]
    except Exception:
        return "unknown"

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

def run_pipeline():
    df = load_data(r"D:\MLOPS\data\raw\AirPassengers.csv")

    metadata = {
        "git_commit": get_git_commit(),
        "dvc_data_hash": get_dvc_data_hash(r"data/raw/AirPassengers.csv.dvc")
    }

    for window in [12, 24, 36]:
        with mlflow.start_run():
            metadata["forecast_window"] = window
            model, rmse = train_and_evaluate(df, window)
            log_experiment(model, rmse, metadata)
            print(f"Window={window} | RMSE={rmse}")

if __name__ == "__main__":
    run_pipeline()
