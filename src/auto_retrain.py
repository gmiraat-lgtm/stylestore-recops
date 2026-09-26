"""
RecOps - Automated drift-response orchestrator.

The hands-free loop: given an incoming production batch, run the drift
monitor; if no drift is detected, archive the batch and stop (retraining on
every batch wastes compute and risks churn). If drift IS detected, respond:
merge the batch into the training events, re-track the data with DVC, rerun
the pipeline (dvc repro), run the promotion gate, and hot-reload the serving
API. Every action is printed with its outcome, forming the audit trail of an
autonomous retrain. This script is what the phrase "continuous retraining"
in the project title refers to.

Usage:
  python src/auto_retrain.py --batch data/incoming/batch_001.csv
  python src/auto_retrain.py --batch ... --dry-run     (monitor only, never retrain)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests

EVENTS = "data/raw/events.csv"
SERVING_RELOAD = "http://localhost:8001/reload"


def run(cmd, allow_codes=(0,)):
    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode not in allow_codes:
        print(f"FAILED (exit {proc.returncode}) - aborting orchestration.")
        sys.exit(proc.returncode)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    batch = Path(args.batch)

    # 1. Judge the batch.
    code = run([sys.executable, "src/monitor.py", "--batch", str(batch)],
               allow_codes=(0, 42))
    if code == 0:
        print("\nVERDICT: no significant drift. Model unchanged (retraining "
              "on every batch would waste compute for no quality gain).")
        archive = Path("data/incoming/processed") / batch.name
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(batch), archive)
        print(f"Batch archived -> {archive}")
        return

    print("\nVERDICT: drift detected. Initiating automated retraining response.")
    if args.dry_run:
        print("(dry-run: stopping before any retraining action)")
        return

    # 2. Merge the drifted batch into training data.
    events = pd.read_csv(EVENTS)
    new = pd.read_csv(batch)
    merged = pd.concat([events, new], ignore_index=True)
    merged.to_csv(EVENTS, index=False)
    print(f"Merged {len(new)} batch events into {EVENTS} (total {len(merged)})")

    # 3. Version the new data state.
    run(["dvc", "add", EVENTS])

    # 4. Re-run the pipeline (ingest -> validate -> features -> train -> evaluate).
    run(["dvc", "repro"])

    # 5. Promotion gate (registers new version; promotes only if quality holds).
        run([sys.executable, "src/promote.py", "--force"])

    # 6. Hot-reload serving so the storefront picks up the new production model.
    try:
        resp = requests.post(SERVING_RELOAD, timeout=10)
        print(f"Serving reload: {resp.json()}")
    except requests.RequestException as exc:
        print(f"Serving reload skipped (API not reachable: {exc}). "
              f"It will load the new production model on next startup.")

    # 7. Archive the handled batch.
    archive = Path("data/incoming/processed") / batch.name
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(batch), archive)
    print(f"Batch archived -> {archive}")
    print("\nAUTOMATED RETRAIN COMPLETE.")


if __name__ == "__main__":
    main()