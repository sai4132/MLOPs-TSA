# src/train_flow.py

from prefect import flow
from train import train_pipeline


@flow(name="tsa-training-pipeline")
def orchestrated_training():
    """
    Prefect control plane:
    - scheduling
    - retries
    - visibility
    """
    train_pipeline()


if __name__ == "__main__":
    orchestrated_training()
