"""
RecOps pipeline - Stage 3: build_features.

Transforms the validated event stream into the user-item interaction matrix
that the recommender trains on. Every (user, product) pair is scored by
summing action weights from params.yaml, so stronger funnel signals
(add_to_cart, purchase) contribute more preference strength than views.
The resulting interactions.csv is the project's lightweight feature store:
a versioned, reusable feature table that any model head can consume.
"""

from pathlib import Path

import pandas as pd
import yaml


def main():
    params = yaml.safe_load(open("params.yaml"))["features"]

    events = pd.read_csv(params["input"])
    weights = params["action_weights"]

    events["weight"] = events["action"].map(weights)

    interactions = (
        events.groupby(["user_id", "product_id"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "strength"})
    )

    out = Path(params["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(out, index=False)

    print(f"Features: {len(events)} events -> {len(interactions)} user-item interactions")
    print(f"Users: {interactions['user_id'].nunique()}, "
          f"Items: {interactions['product_id'].nunique()}")
    print(f"Strength stats:\n{interactions['strength'].describe()}")


if __name__ == "__main__":
    main()