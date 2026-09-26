"""
RecOps - Model serving API (Flask) with A/B champion-challenger serving.

Since Phase 4b the service loads TWO models from the registry: the
'production' alias (champion) and, when one exists, the 'challenger' alias.
Each user is deterministically assigned to a variant by hashing their user
id - the same user always gets the same model, which is what makes the
experiment interpretable. Every /recommend response is logged as an
impression for its variant, and the storefront reports clicks on
recommended products back to /ab/click, so the two models are compared on
LIVE engagement (click-through rate) rather than offline metrics. /ab/report
summarises the experiment; /ops endpoints and the Report Center remain as
before.

Run:  python serving/app.py   (port 8001)
"""

import csv
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
ROOT = Path(__file__).resolve().parent.parent
AB_LOG = ROOT / "reports" / "generated" / "ab_events.csv"

app = Flask(__name__)
state = {
    "champion": {"model": None, "version": None},
    "challenger": {"model": None, "version": None},
    "loaded_at": None,
}


def load_alias(client, alias):
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, alias)
    except Exception:
        return None, None
    local_path = download_artifacts(artifact_uri=mv.source)
    return load_model(local_path), mv.version


def load_models():
    client = MlflowClient()
    champ_model, champ_v = load_alias(client, "production")
    chall_model, chall_v = load_alias(client, "challenger")
    state["champion"] = {"model": champ_model, "version": champ_v}
    state["challenger"] = {"model": chall_model, "version": chall_v}
    state["loaded_at"] = datetime.now().isoformat(timespec="seconds")
    print(f"Champion:   v{champ_v}")
    print(f"Challenger: v{chall_v if chall_v else '- (none registered)'}")


def assign_variant(user_id: str) -> str:
    """Deterministic 50/50 split; without a challenger everyone is champion."""
    if not state["challenger"]["model"]:
        return "champion"
    h = 0
    for c in user_id:
        h = (h * 31 + ord(c)) % (2 ** 32)
    return "challenger" if h % 2 else "champion"


def log_ab(event_type, user_id, variant, version, product_id=""):
    AB_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not AB_LOG.exists()
    with open(AB_LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "event", "user_id", "variant", "model_version", "product_id"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    event_type, user_id, variant, version, product_id])


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "champion_version": state["champion"]["version"],
        "challenger_version": state["challenger"]["version"],
        "ab_active": state["challenger"]["model"] is not None,
    })


@app.get("/recommend/<user_id>")
def recommend(user_id):
    n = request.args.get("n", default=10, type=int)
    variant = assign_variant(user_id)
    slot = state[variant]
    recs = recommend_for_user(slot["model"], user_id, n=n)
    if recs:
        log_ab("impression", user_id, variant, slot["version"])
    return jsonify({
        "user_id": user_id,
        "variant": variant,
        "model_version": slot["version"],
        "recommendations": [{"product_id": pid, "score": round(score, 4)} for pid, score in recs],
        "cold_start": len(recs) == 0,
    })


@app.post("/ab/click")
def ab_click():
    data = request.get_json(force=True, silent=True) or {}
    user_id = str(data.get("user_id", ""))
    product_id = str(data.get("product_id", ""))
    variant = assign_variant(user_id)
    log_ab("click", user_id, variant, state[variant]["version"], product_id)
    return jsonify({"logged": True, "variant": variant})


def ab_summary():
    counts = {"champion": {"impressions": 0, "clicks": 0},
              "challenger": {"impressions": 0, "clicks": 0}}
    if AB_LOG.exists():
        with open(AB_LOG, newline="") as f:
            for row in csv.DictReader(f):
                v = row["variant"]
                if v in counts:
                    key = "impressions" if row["event"] == "impression" else "clicks"
                    counts[v][key] += 1
    for v, c in counts.items():
        c["ctr"] = round(c["clicks"] / c["impressions"], 4) if c["impressions"] else None
        c["model_version"] = state[v]["version"]
    return counts


@app.get("/ab/report")
def ab_report():
    return jsonify(ab_summary())


@app.post("/reload")
def reload_models():
    old = {v: state[v]["version"] for v in ("champion", "challenger")}
    load_models()
    return jsonify({"reloaded": True, "old": old,
                    "new": {v: state[v]["version"] for v in ("champion", "challenger")}})


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return max(sum(1 for _ in f) - 1, 0)


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
            "production_version": state["champion"]["version"],
            "challenger_version": state["challenger"]["version"],
            "ab_active": state["challenger"]["model"] is not None,
            "loaded_at": state["loaded_at"],
            "users_known": len(state["champion"]["model"]["profiles"])
                           if state["champion"]["model"] else 0,
        },
        "data": {
            "training_events": count_lines(ROOT / "data" / "raw" / "events.csv"),
            "pending_batches": incoming,
            "processed_batches": processed,
        },
        "last_drift_check": drift,
        "ab": ab_summary(),
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
            "is_production": str(mv.version) == str(state["champion"]["version"]),
            "is_challenger": str(mv.version) == str(state["challenger"]["version"]),
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
    os.chdir(ROOT)
    load_models()
    app.run(host="0.0.0.0", port=8001)
