import yaml
import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from datetime import datetime
from config import MODEL_REGISTRY_PATH, LOG_DIR

# -----------------------
# App
# -----------------------

app = FastAPI(title="TSA Forecasting Service")

model = None
model_metadata = None


# -----------------------
# Registry + model loading
# -----------------------

def load_registry(path=MODEL_REGISTRY_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_production_model():
    global model, model_metadata

    registry = load_registry()
    prod = registry.get("production")

    if not prod or not prod.get("run_id"):
        raise RuntimeError("No production model found in registry")

    model_uri = f"runs:/{prod['run_id']}/{prod['model_path']}"
    model = mlflow.pyfunc.load_model(model_uri)
    model_metadata = prod

def log_inference(input_t, prediction, metadata):
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = f"{LOG_DIR}/inference_log.csv"

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_t": input_t,
        "prediction": prediction,
        "run_id": metadata["run_id"],
        "git_commit": metadata["git_commit"],
        "dvc_data_hash": metadata["dvc_data_hash"],
    }

    df = pd.DataFrame([row])

    if os.path.exists(log_path):
        df.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(log_path, index=False)


# -----------------------
# Startup hook
# -----------------------

@app.on_event("startup")
def startup_event():
    load_production_model()
    print("Production model loaded")


# -----------------------
# API schemas
# -----------------------

class ForecastRequest(BaseModel):
    t: int


class ForecastResponse(BaseModel):
    prediction: float


# -----------------------
# Endpoints
# -----------------------

@app.get("/")
def health():
    return {
        "status": "ok",
        "rmse": model_metadata["rmse"],
        "git_commit": model_metadata["git_commit"],
        "dvc_data_hash": model_metadata["dvc_data_hash"],
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        df = pd.DataFrame({"t": [req.t]})
        pred = model.predict(df)[0]
        log_inference(
            input_t=req.t,
            prediction=float(pred),
            metadata=model_metadata,
        )
        return ForecastResponse(prediction=float(pred))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))