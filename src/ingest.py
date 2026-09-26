"""
RecOps pipeline - Stage 1: ingest.

Reads raw interaction events and the product catalog, keeps only events that
reference real catalog products, parses timestamps, and writes a clean sorted
event stream for the rest of the pipeline. This is the single entry point for
data into the system, so every later stage can trust product_id integrity.
"""

from pathlib import Path

import pandas as pd
import yaml


def main():
    params = yaml.safe_load(open("params.yaml"))["ingest"]

    events = pd.read_csv(params["events_path"])
    catalog = pd.read_csv(params["catalog_path"])
    n_raw = len(events)

    valid_ids = set(catalog["product_id"])
    events = events[events["product_id"].isin(valid_ids)].copy()

    events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce")
    events = events.dropna(subset=["timestamp"]).sort_values("timestamp")

    out = Path(params["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(out, index=False)
    print(f"Ingest: {n_raw} raw events -> {len(events)} kept -> {out}")


if __name__ == "__main__":
    main()
