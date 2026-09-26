"""
RecOps - Model serving API (Flask).

Serves the production recommender over HTTP. On startup (and on /reload) the
service asks the MLflow Model Registry which model version currently holds
the 'production' alias, downloads that artifact, and serves it. The API never
references a file path directly - promotion in the registry is the single
switch that changes what is served, which is what makes automated
retrain-and-promote invisible to API consumers.

Endpoints:
  GET /health                       liveness + which model version is loaded
  GET /recommend/<user_id>?n=10     top-n recommendations for a user
  POST /reload                      re-resolve the production alias and reload

Run:  python serving/app.py   (listens on port 8001; StyleStore owns 8000)
"""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recommender import load_model, recommend_for_user  # noqa: E402

MODEL_NAME = "recops-item-cf"
ALIAS = "production"

app = Flask(__name__)
state = {"model": None, "version": None}


def load_production_model():
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    local_path = download_artifacts(artifact_uri=mv.source)
    state["model"] = load_model(local_path)
    state["version"] = mv.version
    print(f"Loaded {MODEL_NAME} v{mv.version} ({ALIAS}) from {mv.source}")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "version": state["version"],
        "users_known": len(state["model"]["profiles"]) if state["model"] else 0,
    })


@app.get("/recommend/<user_id>")
def recommend(user_id):
    n = request.args.get("n", default=10, type=int)
    recs = recommend_for_user(state["model"], user_id, n=n)
    return jsonify({
        "user_id": user_id,
        "model_version": state["version"],
        "recommendations": [{"product_id": pid, "score": round(score, 4)} for pid, score in recs],
        "cold_start": len(recs) == 0,
    })


@app.post("/reload")
def reload_model():
    old = state["version"]
    load_production_model()
    return jsonify({"reloaded": True, "old_version": old, "new_version": state["version"]})


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)  # so mlruns/ resolves
    load_production_model()
    app.run(host="0.0.0.0", port=8001)