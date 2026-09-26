"""
RecOps - Model serving API (Flask) with operations endpoints.

Serves the production recommender over HTTP and, since Phase 4, also exposes
the operational state of the whole MLOps system so the admin dashboard can
render it: which model version is live, the full registry version history
with metrics, event-log size, and the latest drift verdict. Also serves the
generated Report Center documents for download. The model itself is resolved
through the MLflow Model Registry 'production' alias, so promotion remains
the single switch that changes what is served.

Endpoints:
  GET /health                       liveness + which model version is loaded
  GET /recommend/<user_id>?n=10     top-n recommendations for a user
  POST /reload                      re-resolve the production alias and reload
  GET /ops/status                   system snapshot for the dashboard
  GET /ops/versions                 registry history with metrics per version
  GET /reports/<file>               download a generated Report Center file

Run:  python serving/app.py   (listens on port 8001; StyleStore owns 8000)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recommender import load_model, recommend_for_user  # noqa: E402

MODEL_NAME = "recops-item-cf"
ALIAS = "production"
ROOT = Path(__file__).resolve().parent.parent

app = Flask(__name__)
state = {"model": None, "version": None, "loaded_at": None}


def load_production_model():
    client = MlflowClient()
    mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    local_path = download_artifacts(artifact_uri=mv.source)
    state["model"] = load_model(local_path)
    state["version"] = mv.version
    state["loaded_at"] = datetime.now().isoformat(timespec="seconds")
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


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus header


@app.get("/ops/status")
def ops_status():
    drift_path = ROOT / "reports" / "drift" / "drift_status.json"
    drift = json.loads(drift_path.read_text()) if drift_path.exists() else None
    incoming = sorted(p.name for p in (ROOT / "data" / "incoming").glob("batch_*.csv"))
    processed = sorted(p.name for p in
                       (ROOT / "data" / "incoming" / "processed").glob("batch_*.csv"))
    return jsonify({
        "serving": {
            "model": MODEL_NAME,
            "production_version": state["version"],
            "loaded_at": state["loaded_at"],
            "users_known": len(state["model"]["profiles"]) if state["model"] else 0,
        },
        "data": {
            "training_events": count_lines(ROOT / "data" / "raw" / "events.csv"),
            "pending_batches": incoming,
            "processed_batches": processed,
        },
        "last_drift_check": drift,
    })


@app.get("/ops/versions")
def ops_versions():
    client = MlflowClient()
    out = []
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        run = client.get_run(mv.run_id)
        m = run.data.metrics
        out.append({
            "version": int(mv.version),
            "run_id": mv.run_id,
            "created": datetime.fromtimestamp(mv.creation_timestamp / 1000)
                       .strftime("%Y-%m-%d %H:%M"),
            "is_production": str(mv.version) == str(state["version"]),
            "precision_at_10": m.get("precision_at_10"),
            "recall_at_10": m.get("recall_at_10"),
            "category_relevance_at_10": m.get("category_relevance_at_10"),
            "train_seconds": m.get("train_seconds"),
        })
    out.sort(key=lambda v: v["version"], reverse=True)
    return jsonify(out)


@app.get("/reports/<path:name>")
def reports(name):
    for base in (ROOT / "reports" / "generated", ROOT / "reports" / "drift"):
        if (base / name).exists():
            return send_from_directory(base, name)
    return jsonify({"error": f"report '{name}' not found"}), 404


if __name__ == "__main__":
    os.chdir(ROOT)  # so mlruns/ resolves
    load_production_model()
    app.run(host="0.0.0.0", port=8001)
