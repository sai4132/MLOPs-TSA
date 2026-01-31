import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
MODEL_REGISTRY_PATH = os.getenv("MODEL_REGISTRY_PATH", "model_registry.yaml")
LOG_DIR = os.getenv("LOG_DIR", "logs")
