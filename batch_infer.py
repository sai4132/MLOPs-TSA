import yaml
import mlflow
import pandas as pd

# -----------------------
# Load registry
# -----------------------

def load_registry(path="model_registry.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

registry = load_registry()
prod = registry.get("production")

if not prod or not prod.get("run_id"):
    raise RuntimeError("No production model found in registry")

# -----------------------
# Load model
# -----------------------

model_uri = f"runs:/{prod['run_id']}/{prod['model_path']}"
model = mlflow.pyfunc.load_model(model_uri)

# -----------------------
# Batch inference
# -----------------------

def run_batch_inference(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    if "t" not in df.columns:
        raise ValueError("Input CSV must contain column 't'")

    preds = model.predict(df)
    df["prediction"] = preds

    df.to_csv(output_csv, index=False)

    print(f"Batch inference complete → {output_csv}")
    print(f"Model RMSE: {prod['rmse']}")
    print(f"Git commit: {prod['git_commit']}")
    print(f"DVC data hash: {prod['dvc_data_hash']}")

# -----------------------
# Entry point
# -----------------------

if __name__ == "__main__":
    run_batch_inference(
        input_csv="batch_input.csv",
        output_csv="batch_output.csv",
    )