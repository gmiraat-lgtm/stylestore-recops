"""
RecOps - Synthetic user interaction event simulator.

Generates realistic e-commerce interaction events (view / click / add_to_cart /
purchase) against the real StyleStore product catalog. Users are assigned to
behavioural personas, and each persona favours particular category/gender
segments of the catalog. Within every persona segment, item popularity follows
a Zipf distribution - a few bestsellers attract most interactions, exactly as
in real e-commerce - hence the co-occurrence structure that collaborative
filtering learns from actually exists in the data. Because the persona mix is
passed on the command line, the distribution of incoming events can be shifted
between runs - that is why this simulator doubles as a controllable drift
generator for Phase 2.

Usage:
  python src/simulate_events.py --users 2000 --events 200000 --days 30
  python src/simulate_events.py --mix "ethnic:0.5,formal:0.3,accessories:0.2" ...
"""

import argparse
import itertools
import random
from datetime import datetime, timedelta

import pandas as pd

PERSONAS = {
    "sneakerhead":  {"gender": ["Men", "Unisex"],
                     "categories": ["Casual Shoes", "Sports Shoes", "Tshirts", "Flip Flops"]},
    "ethnic":       {"gender": ["Women"],
                     "categories": ["Kurtas", "Sandals", "Heels", "Handbags"]},
    "western":      {"gender": ["Women"],
                     "categories": ["Tops", "Tshirts", "Heels", "Handbags", "Perfume and Body Mist"]},
    "formal":       {"gender": ["Men"],
                     "categories": ["Shirts", "Watches", "Wallets", "Belts"]},
    "accessories":  {"gender": ["Men", "Women", "Unisex"],
                     "categories": ["Watches", "Sunglasses", "Wallets", "Perfume and Body Mist"]},
}

DEFAULT_MIX = "sneakerhead:0.25,ethnic:0.25,western:0.20,formal:0.15,accessories:0.15"

ACTIONS = ["view", "click", "add_to_cart", "purchase"]
ACTION_WEIGHTS = [0.62, 0.22, 0.10, 0.06]

EXPLORATION_RATE = 0.15  # chance of an event outside the persona's segment


def parse_mix(mix_str):
    mix = {}
    for part in mix_str.split(","):
        name, weight = part.split(":")
        name = name.strip()
        if name not in PERSONAS:
            raise ValueError(f"Unknown persona '{name}'. Valid: {list(PERSONAS)}")
        mix[name] = float(weight)
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def build_pools(catalog, zipf_s):
    """
    For each persona, build (item_list, cumulative_weights) where the weight of
    the item at popularity rank r is 1/(r+1)^s. The popularity ranking itself is
    a seeded shuffle of the segment, so it is stable for a given --seed.
    """
    pools = {}
    for name, spec in PERSONAS.items():
        subset = catalog[
            catalog["gender"].isin(spec["gender"])
            & catalog["category"].isin(spec["categories"])
        ]
        ids = subset["product_id"].tolist()
        if not ids:
            raise ValueError(f"Persona '{name}' matched no products - check categories.")
        random.shuffle(ids)
        weights = [1.0 / (rank + 1) ** zipf_s for rank in range(len(ids))]
        pools[name] = (ids, list(itertools.accumulate(weights)))
    return pools


def main():
    ap = argparse.ArgumentParser(description="RecOps synthetic event generator")
    ap.add_argument("--users", type=int, default=2000)
    ap.add_argument("--events", type=int, default=200000)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--end-date", default="2026-09-26", help="Last day of the window (YYYY-MM-DD)")
    ap.add_argument("--mix", default=DEFAULT_MIX, help="Persona mix, e.g. 'ethnic:0.4,formal:0.6'")
    ap.add_argument("--zipf", type=float, default=1.0,
                    help="Zipf exponent for item popularity inside each persona segment")
    ap.add_argument("--out", default="data/raw/events.csv")
    ap.add_argument("--users-out", default="data/raw/users.csv",
                    help="Ground-truth user->persona map (validation only, never for training)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true",
                    help="Append to existing events file instead of overwriting (drift batches)")
    args = ap.parse_args()

    random.seed(args.seed)

    catalog = pd.read_csv("data/raw/products.csv")
    pools = build_pools(catalog, args.zipf)
    all_products = catalog["product_id"].tolist()
    mix = parse_mix(args.mix)

    persona_names = list(mix.keys())
    persona_weights = list(mix.values())
    users = {
        f"u{uid:05d}": random.choices(persona_names, weights=persona_weights, k=1)[0]
        for uid in range(1, args.users + 1)
    }

    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    start = end - timedelta(days=args.days)
    window_seconds = int((end - start).total_seconds())

    rows = []
    user_ids = list(users.keys())
    for _ in range(args.events):
        user_id = random.choice(user_ids)
        persona = users[user_id]

        if random.random() < EXPLORATION_RATE:
            product_id = random.choice(all_products)
        else:
            ids, cum = pools[persona]
            product_id = random.choices(ids, cum_weights=cum, k=1)[0]

        action = random.choices(ACTIONS, weights=ACTION_WEIGHTS, k=1)[0]
        ts = start + timedelta(seconds=random.randint(0, window_seconds))
        rows.append((user_id, product_id, action, ts.strftime("%Y-%m-%d %H:%M:%S")))

    events = pd.DataFrame(rows, columns=["user_id", "product_id", "action", "timestamp"])
    events = events.sort_values("timestamp").reset_index(drop=True)

    if args.append:
        old = pd.read_csv(args.out)
        events = pd.concat([old, events], ignore_index=True)

    events.to_csv(args.out, index=False)
    pd.DataFrame(users.items(), columns=["user_id", "persona"]).to_csv(args.users_out, index=False)

    print(f"Wrote {len(events)} events -> {args.out}")
    print(f"Wrote {len(users)} users  -> {args.users_out}")
    print("\nAction distribution:")
    print(events["action"].value_counts())
    print("\nPersona mix used:", {k: round(v, 2) for k, v in mix.items()})


if __name__ == "__main__":
    main()
