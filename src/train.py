import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import subprocess
import yaml

# ---------- helpers ----------
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

# ---------- data ----------
df = pd.read_csv(r"D:\MLOPS\data\raw\AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df["t"] = np.arange(len(df))

window_sizes = [12, 24, 36]

git_commit = get_git_commit()
data_hash = get_dvc_data_hash(r"D:\MLOPS\data\raw\AirPassengers.csv.dvc")

for window in window_sizes:
    with mlflow.start_run():
        train = df.iloc[:-window]
        test = df.iloc[-window:]

        X_train = train[["t"]]
        y_train = train["Passengers"]
        X_test = test[["t"]]
        y_test = test["Passengers"]

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # ---- log metadata ----
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("forecast_window", window)
        mlflow.log_param("git_commit", git_commit)
        mlflow.log_param("dvc_data_hash", data_hash)

        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Window={window} | RMSE={rmse}")
