"""
RecOps - Model serving API (container variant).

Same endpoints as serving/app.py, but built for the immutable-image pattern:
the model artifact is baked into the Docker image at build time and loaded
from a fixed path, so the container needs no MLflow registry connection at
runtime. The served model version is passed in as a build argument and
reported by /health, keeping version traceability without the registry.
There is no /reload here by design - a new model means a new image.
"""

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recommender import load_model, recommend_for_user  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "item_cf.pkl"
MODEL_VERSION = os.environ.get("MODEL_VERSION", "dev")

app = Flask(__name__)
model = load_model(str(MODEL_PATH))
print(f"Loaded baked-in model ({MODEL_VERSION}) from {MODEL_PATH}")


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": "recops-item-cf",
        "version": MODEL_VERSION,
        "users_known": len(model["profiles"]),
        "mode": "container-immutable",
    })


@app.get("/recommend/<user_id>")
def recommend(user_id):
    n = request.args.get("n", default=10, type=int)
    recs = recommend_for_user(model, user_id, n=n)
    return jsonify({
        "user_id": user_id,
        "model_version": MODEL_VERSION,
        "recommendations": [{"product_id": pid, "score": round(score, 4)} for pid, score in recs],
        "cold_start": len(recs) == 0,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
