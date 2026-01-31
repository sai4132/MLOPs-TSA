import yaml
import mlflow
import pandas as pd
from prefect import flow, task
import os
from datetime import datetime


# -----------------------
# Tasks
# -----------------------

@task
def load_registry(path="model_registry.yaml"):
    with open(path, "r") as f:
        registry = yaml.safe_load(f)

    prod = registry.get("production")
    if not prod or not prod.get("run_id"):
        raise RuntimeError("No production model found in registry")

    return prod


@task
def load_model(prod):
    model_uri = f"runs:/{prod['run_id']}/{prod['model_path']}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


@task(retries=2, retry_delay_seconds=5)
def load_batch_input(input_csv):
    df = pd.read_csv(input_csv)
    if "t" not in df.columns:
        raise ValueError("Input CSV must contain column 't'")
    return df


@task
def run_predictions(model, df):
    preds = model.predict(df)
    df["prediction"] = preds
    return df


@task
def save_output(df, output_csv):
    df.to_csv(output_csv, index=False)
    return len(df)


@task
def log_metadata(prod):
    print("Batch inference metadata:")
    print(f"  RMSE: {prod['rmse']}")
    print(f"  Git commit: {prod['git_commit']}")
    print(f"  DVC data hash: {prod['dvc_data_hash']}")

@task
def log_batch_inference(
    prod,
    input_csv,
    output_csv,
    num_rows,
):
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/batch_inference_log.csv"

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_csv": input_csv,
        "output_csv": output_csv,
        "num_rows": num_rows,
        "run_id": prod["run_id"],
        "git_commit": prod["git_commit"],
        "dvc_data_hash": prod["dvc_data_hash"],
    }

    df = pd.DataFrame([row])

    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)



# -----------------------
# Flow
# ---------------------

@flow(name="batch-inference-pipeline")
def batch_inference_flow(
    input_csv="batch_input.csv",
    output_csv="batch_output.csv",
):
    prod = load_registry()
    model = load_model(prod)
    df = load_batch_input(input_csv)
    df = run_predictions(model, df)
    num_rows = save_output(df, output_csv)
    log_batch_inference(
        prod=prod,
        input_csv=input_csv,
        output_csv=output_csv,
        num_rows=num_rows,
    )
    log_metadata(prod)


if __name__ == "__main__":
    batch_inference_flow()