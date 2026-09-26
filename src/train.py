"""
RecOps pipeline - Stage 4: train.

Trains the item-based CF recommender on the full interaction matrix and logs
the run to MLflow: hyperparameters, dataset size, and the model artifact.
The MLflow run_id is written to models/run_id.txt so the evaluation stage can
attach its metrics to the same run - one run in the tracking UI then tells
the complete story of one pipeline execution.
"""

import time
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from recommender import build_model, save_model


def main():
    params = yaml.safe_load(open("params.yaml"))["train"]

    interactions = pd.read_csv(params["input"])

    mlflow.set_experiment("recops")
    with mlflow.start_run() as run:
        mlflow.log_params({
            "top_k_neighbors": params["top_k_neighbors"],
            "min_item_interactions": params["min_item_interactions"],
            "n_interactions": len(interactions),
            "n_users": interactions["user_id"].nunique(),
            "n_items_raw": interactions["product_id"].nunique(),
        })

        t0 = time.time()
        model = build_model(
            interactions,
            top_k_neighbors=params["top_k_neighbors"],
            min_item_interactions=params["min_item_interactions"],
        )
        train_seconds = round(time.time() - t0, 2)

        out = Path(params["model_out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, str(out))

        mlflow.log_metrics({
            "train_seconds": train_seconds,
            "n_items_kept": model["n_items"],
        })
        mlflow.log_artifact(str(out))

        Path("models/run_id.txt").write_text(run.info.run_id)
        print(f"Train: {model['n_users']} users x {model['n_items']} items "
              f"in {train_seconds}s -> {out}")
        print(f"MLflow run: {run.info.run_id}")


if __name__ == "__main__":
    main()