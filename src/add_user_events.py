import argparse
import random
from datetime import datetime, timedelta

import pandas as pd

from simulate_events import PERSONAS

ACTIONS = ["view", "click", "add_to_cart", "purchase"]
ACTION_WEIGHTS = [0.55, 0.25, 0.12, 0.08]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--persona", required=True, choices=list(PERSONAS))
    ap.add_argument("--events", type=int, default=800)
    ap.add_argument("--top", type=int, default=400,
                    help="Sample only from this many most-popular pool items")
    ap.add_argument("--events-file", default="data/raw/events.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    random.seed(args.seed)
    catalog = pd.read_csv("data/raw/products.csv")
    spec = PERSONAS[args.persona]
    pool = set(catalog[
        catalog["gender"].isin(spec["gender"])
        & catalog["category"].isin(spec["categories"])
    ]["product_id"])

    events = pd.read_csv(args.events_file)
    counts = events[events["product_id"].isin(pool)]["product_id"].value_counts()
    popular = counts.head(args.top)
    items = popular.index.tolist()
    weights = popular.tolist()

    now = datetime.now()
    rows = []
    for _ in range(args.events):
        rows.append((
            args.user,
            random.choices(items, weights=weights, k=1)[0],
            random.choices(ACTIONS, weights=ACTION_WEIGHTS, k=1)[0],
            (now - timedelta(seconds=random.randint(0, 86400))).strftime("%Y-%m-%d %H:%M:%S"),
        ))

    new = pd.DataFrame(rows, columns=["user_id", "product_id", "action", "timestamp"])
    out = pd.concat([events, new], ignore_index=True)
    out.to_csv(args.events_file, index=False)
    print(f"Appended {len(new)} popularity-weighted '{args.persona}' events for {args.user} "
          f"(top {len(items)} items) -> {args.events_file} (total {len(out)})")


if __name__ == "__main__":
    main()
