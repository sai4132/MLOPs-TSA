import yaml
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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
# Load model from MLflow
# -----------------------

model_uri = f"runs:/{prod['run_id']}/{prod['model_path']}"
model = mlflow.pyfunc.load_model(model_uri)

# -----------------------
# FastAPI app
# -----------------------

app = FastAPI(title="TSA Forecasting Service")

class ForecastRequest(BaseModel):
    t: int  # time index

class ForecastResponse(BaseModel):
    prediction: float

@app.get("/")
def health():
    return {
        "status": "ok",
        "rmse": prod["rmse"],
        "git_commit": prod["git_commit"],
        "dvc_data_hash": prod["dvc_data_hash"],
    }

@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    try:
        df = pd.DataFrame({"t": [req.t]})
        pred = model.predict(df)[0]
        return ForecastResponse(prediction=float(pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
