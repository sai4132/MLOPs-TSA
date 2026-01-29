# Time Series MLOps Pipeline (Training → Registry → Serving)

This repository implements an **end-to-end MLOps system for time-series forecasting**, covering:

- data versioning
- experiment tracking
- automated training & evaluation
- model promotion via a registry
- real-time serving
- batch inference

The project is intentionally built using **lightweight, transparent components** to clearly demonstrate MLOps concepts rather than hiding logic behind managed platforms.

---

## System Overview

The system is split into **offline (training)** and **online (serving/inference)** components, connected via a **model registry contract**.

### High-level flow

---

## Core Design Principles

- **Separation of concerns**
  - Training decides
  - Serving only consumes decisions
- **Reproducibility**
  - Code → Git
  - Data → DVC
  - Experiments → MLflow
- **Explicit contracts**
  - `model_registry.yaml` is the single source of truth
- **Deterministic behavior**
  - Same code + data → same model
  - No silent promotions

---

## Repository Structure

---

## Tools Used (and Why)

| Tool | Purpose |
|-----|--------|
| Git | Code versioning |
| DVC | Data versioning (content-addressed) |
| MLflow | Experiment tracking & model artifacts |
| Prefect | Pipeline orchestration |
| FastAPI | Real-time model serving |
| uv | Deterministic Python environment |

---

## Training Pipeline (`main.py`)

### What it does

- Loads versioned time-series data
- Trains multiple models (different forecast windows)
- Evaluates using RMSE
- Applies a **quality gate**
- Promotes the best model to production
- Updates the model registry

### Key properties

- Fully reproducible
- Deterministic
- Promotion is **conditional**, not automatic
- Logs:
  - Git commit hash
  - DVC data hash
  - Metrics
  - Promotion status

### Run training

```bash
python src/train.py
mlflow ui
prefect server start

### Start Service
uvicorn serve:app --reload

