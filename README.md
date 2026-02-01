End-to-End MLOps Pipeline (Time Series Forecasting)

This repository implements a production-style MLOps pipeline for time-series forecasting, covering training, experiment tracking, model promotion, and Dockerized serving.

The project is designed to expose real system boundaries (artifacts, registries, mounts) instead of hiding them behind abstractions.

What This Project Does

Trains time-series models and tracks experiments using MLflow

Versions data using DVC

Promotes the best model via an explicit file-based model registry

Serves the production model using FastAPI

Runs training and serving inside Docker containers

Supports batch inference using the same production model

High-Level Flow
Data (DVC)
   ↓
Training (Docker)
   ↓
MLflow Artifacts (mlruns/)
   ↓
model_registry.yaml (production pointer)
   ↓
Serving (FastAPI + Docker)


Key principle:

Runs are history. Models are deployable assets.
Serving always loads from the registry, not from “latest runs”.

Project Structure
.
├── src/
│   ├── train.py        # Training execution (no orchestration)
│   ├── train_flow.py   # Optional Prefect wrapper (control plane)
│
├── serve.py            # FastAPI app for real-time inference
│
├── Dockerfile.train    # Training container
├── Dockerfile.serve    # Serving container
│
├── data/               # DVC-tracked data
├── mlruns/             # MLflow artifacts (not committed)
├── logs/               # Training & inference logs
│
├── model_registry.yaml # File-based production model registry
│
├── batch_infer.py      # Batch inference
├── batch_pipeline.py
│
├── config.py
├── pyproject.toml
├── uv.lock
├── README.md

Model Registry

model_registry.yaml is the single source of truth for serving.

Example:

production:
  model_uri: "/app/mlruns/0/models/m-xxxx/artifacts"
  rmse: 39.93
  git_commit: a91f2e1
  dvc_data_hash: 3b2c4a


Stores an explicit MLflow artifact path

No run IDs

No implicit assumptions

Serving fails fast if the registry is invalid (by design)

Training
Local
python src/train.py

Dockerized
docker build -f Dockerfile.train -t tsa-train .

docker run \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/model_registry.yaml:/app/model_registry.yaml \
  -v $(pwd)/data:/app/data \
  tsa-train


Training:

logs experiments to MLflow

applies a quality gate

promotes the best model

updates model_registry.yaml

Serving (Real-Time Inference)
docker build -f Dockerfile.serve -t tsa-serve .

docker run -p 8000:8000 \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/model_registry.yaml:/app/model_registry.yaml \
  tsa-serve


API available at: http://localhost:8000/docs

Loads the production model from the registry at startup

Batch Inference
python batch_infer.py


Uses the same production model defined in model_registry.yaml.