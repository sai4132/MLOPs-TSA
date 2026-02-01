import pandas as pd
import mlflow
import mlflow.sklearn
from config import AIR_PASSENGERS_CSV, AIR_PASSENGERS_DVC, REGISTRY_PATH

# src/train.py

from pathlib import Path
import mlflow
import pandas as pd

from config import (
    AIR_PASSENGERS_CSV,
    AIR_PASSENGERS_DVC,
    REGISTRY_PATH,
)

# ---- helpers (replace with your actual implementations if names differ) ----

def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def get_git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

def get_dvc_data_hash(dvc_file: Path) -> str:
    import yaml
    with open(dvc_file, "r") as f:
        return yaml.safe_load(f)["outs"][0]["md5"]

def train_and_evaluate(df: pd.DataFrame, window: int):
    """
    Dummy placeholder.
    Replace with your real training logic.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import root_mean_squared_error
    import numpy as np

    X = np.arange(len(df)).reshape(-1, 1)
    y = df.iloc[:, 1].values

    model = LinearRegression().fit(X[:-window], y[:-window])
    preds = model.predict(X[-window:])
    rmse = root_mean_squared_error(y[-window:], preds)
    return model, rmse

def load_registry():
    import yaml
    if not REGISTRY_PATH.exists():
        return {"production": None}
    with open(REGISTRY_PATH, "r") as f:
        return yaml.safe_load(f)

def save_registry(registry: dict):
    import yaml
    with open(REGISTRY_PATH, "w") as f:
        yaml.safe_dump(registry, f)

def quality_check(rmse: float, threshold: float = 100.0) -> bool:
    return rmse < threshold

# ---------------------------------------------------------------------------


def train_pipeline():
    """
    PURE TRAINING EXECUTION
    - no Prefect
    - no orchestration
    - safe for Docker / CI / batch
    """

    print("Starting training pipeline")

    df = load_data(AIR_PASSENGERS_CSV)
    git_commit = get_git_commit()
    data_hash = get_dvc_data_hash(AIR_PASSENGERS_DVC)

    registry = load_registry()
    current_prod_rmse = (
        registry["production"]["rmse"]
        if registry.get("production")
        else None
    )

    for window in [12, 24, 36]:
        with mlflow.start_run():
            mlflow.log_param("forecast_window", window)
            mlflow.log_param("git_commit", git_commit)
            mlflow.log_param("dvc_data_hash", data_hash)

            model, rmse = train_and_evaluate(df, window)
            mlflow.log_metric("rmse", rmse)
            result = mlflow.sklearn.log_model(
                model,
                artifact_path="model",
            )

            model_uri = result.model_uri

            # normalize for local filesystem use
            if model_uri.startswith("models:/"):
                model_uri = f"/app/mlruns/0/models/{model_uri.split('/')[-1]}/artifacts"

            passed = quality_check(rmse)

            if passed and (current_prod_rmse is None or rmse < current_prod_rmse):
                print(f"Promoting model with RMSE={rmse}")

                registry["production"] = {
                    "model_uri": model_uri,
                    "rmse": float(rmse),
                    "git_commit": git_commit,
                    "dvc_data_hash": data_hash,
                }

                save_registry(registry)
                mlflow.log_param("model_status", "promoted")
                current_prod_rmse = rmse
            else:
                mlflow.log_param("model_status", "not_promoted")

    print("Training pipeline finished")


def main():
    train_pipeline()


if __name__ == "__main__":
    main()
