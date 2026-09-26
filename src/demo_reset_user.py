import random
import sys
from datetime import datetime, timedelta

import pandas as pd

sys.path.insert(0, "src")
from simulate_events import PERSONAS

USER = "u01615"
random.seed(11)

cat = pd.read_csv("data/raw/products.csv")
spec = PERSONAS["sneakerhead"]
pool = set(cat[cat["gender"].isin(spec["gender"])
               & cat["category"].isin(spec["categories"])]["product_id"])

ev = pd.read_csv("data/raw/events.csv")
before = len(ev)
ev = ev[ev["user_id"] != USER]
print(f"removed {before - len(ev)} old {USER} events")

counts = ev[ev["product_id"].isin(pool)]["product_id"].value_counts()
band = counts.iloc[50:500]
items = band.index.tolist()
weights = band.tolist()

now = datetime.now()
rows = [(USER,
         random.choices(items, weights=weights, k=1)[0],
         random.choices(["view", "click", "add_to_cart", "purchase"],
                        weights=[.55, .25, .12, .08], k=1)[0],
         (now - timedelta(seconds=random.randint(0, 86400))).strftime("%Y-%m-%d %H:%M:%S"))
        for _ in range(400)]
ev = pd.concat([ev, pd.DataFrame(rows, columns=ev.columns)], ignore_index=True)
ev.to_csv("data/raw/events.csv", index=False)
print(f"injected 400 mid-popularity sneaker events for {USER}; total {len(ev)}")
