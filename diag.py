import sys
sys.path.insert(0, "src")
import pandas as pd
from recommender import load_model, recommend_for_user

cat = pd.read_csv("data/raw/products.csv").set_index("product_id")["category"]

ev = pd.read_csv("data/raw/events.csv")
mine = ev[ev["user_id"] == "u01615"].copy()
mine["category"] = mine["product_id"].map(cat)
print("=== RAW EVENTS for u01615 ===")
print("total:", len(mine))
print(mine["category"].value_counts().head(8))

m = load_model("models/item_cf.pkl")
prof = m["profiles"].get("u01615", {})
pc = pd.Series({i: cat.get(i) for i in prof}).value_counts()
print("\n=== TRAINED MODEL (models/item_cf.pkl) ===")
print("profile items surviving filter:", len(prof))
print(pc.head(8))

recs = recommend_for_user(m, "u01615", 8)
print("\n=== MODEL'S OWN TOP-8 ===")
for pid, s in recs:
    print(pid, round(s, 2), cat.get(pid))
