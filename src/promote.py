"""
RecOps - Model Registry promotion gate.

Registers the latest trained model (from models/run_id.txt) as a new version
of the 'recops-item-cf' registered model, then decides whether it earns the
'production' alias. The gate compares the candidate's category relevance and
precision against the current production version: promote only if quality has
not regressed. First-ever version is promoted unconditionally (bootstrap).
Run with --force to promote regardless (manual override, still auditable in
the registry history).
"""

import argparse
from pathlib import Path

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

MODEL_NAME = "recops-item-cf"
ALIAS = "production"
# Candidate must retain at least this fraction of production's metric values.
TOLERANCE = 0.95


def get_metric(client, run_id, key):
    run = client.get_run(run_id)
    return run.data.metrics.get(key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Promote regardless of metrics")
    args = ap.parse_args()

    client = MlflowClient()
    run_id = Path("models/run_id.txt").read_text().strip()

    # Ensure the registered model exists (no-op if it already does).
    try:
        client.create_registered_model(MODEL_NAME)
    except MlflowException:
        pass

    # Register the candidate artifact as a new model version.
    run = client.get_run(run_id)
    source = f"{run.info.artifact_uri}/item_cf.pkl"
    version = client.create_model_version(MODEL_NAME, source=source, run_id=run_id).version
    print(f"Registered {MODEL_NAME} version {version} (run {run_id})")

    # Find current production, if any.
    try:
        prod = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except Exception:
        prod = None

    if prod is None or args.force:
        reason = "manual --force" if (prod is not None and args.force) else "first version (bootstrap)"
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
        print(f"PROMOTED v{version} -> '{ALIAS}' ({reason})")
        return

    cand_rel = get_metric(client, run_id, "category_relevance_at_10") or 0.0
    cand_prec = get_metric(client, run_id, "precision_at_10") or 0.0
    prod_rel = get_metric(client, prod.run_id, "category_relevance_at_10") or 0.0
    prod_prec = get_metric(client, prod.run_id, "precision_at_10") or 0.0

    print(f"Candidate v{version}: relevance={cand_rel:.4f}, precision={cand_prec:.4f}")
    print(f"Production v{prod.version}: relevance={prod_rel:.4f}, precision={prod_prec:.4f}")

    if cand_rel >= prod_rel * TOLERANCE and cand_prec >= prod_prec * TOLERANCE:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, version)
        print(f"PROMOTED v{version} -> '{ALIAS}' (quality gate passed)")
    else:
        print(f"NOT promoted: v{version} regresses beyond {int((1-TOLERANCE)*100)}% tolerance. "
              f"v{prod.version} stays in production.")


if __name__ == "__main__":
    main()
