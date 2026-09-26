"""
RecOps pipeline - Stage 5: evaluate.

Offline evaluation of the recommender by per-user holdout: for every user
with enough history, a few of their interacted items are hidden, a model is
rebuilt on the remainder, and we measure (a) how many hidden items appear in
the user's top-k recommendations (precision@k / recall@k), and (b) category
relevance@k - the share of recommended items whose category the user has
already engaged with in training. Category relevance measures whether the
recommendations are personalised to the user's taste segment, which is the
property the drift monitoring in Phase 2 watches degrade. Metrics go to
reports/metrics.json for DVC and onto the SAME MLflow run the training stage
created (via models/run_id.txt).
"""

import json
import random
from pathlib import Path

import mlflow
import pandas as pd
import yaml

from recommender import build_model, recommend_for_user


def main():
    p = yaml.safe_load(open("params.yaml"))
    ev = p["evaluate"]
    tr = p["train"]

    k = ev["k"]
    holdout_n = ev["holdout_per_user"]
    random.seed(42)

    interactions = pd.read_csv(tr["input"])
    catalog = pd.read_csv(p["ingest"]["catalog_path"])
    cat_map = dict(zip(catalog["product_id"], catalog["category"]))

    # Split: hide holdout_n items per eligible user, keep the rest for training.
    test_rows, train_parts = [], []
    for user, group in interactions.groupby("user_id"):
        if len(group) >= holdout_n + 5:
            held = group.sample(n=holdout_n, random_state=42)
            test_rows.append(held)
            train_parts.append(group.drop(held.index))
        else:
            train_parts.append(group)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    model = build_model(
        train_df,
        top_k_neighbors=tr["top_k_neighbors"],
        min_item_interactions=tr["min_item_interactions"],
    )

    user_train_items = train_df.groupby("user_id")["product_id"].apply(set)
    truth = test_df.groupby("user_id")["product_id"].apply(set)

    precisions, recalls, cat_rels = [], [], []
    evaluated = 0
    for user, hidden_items in truth.items():
        recs = recommend_for_user(model, user, n=k)
        if not recs:
            continue
        rec_ids = [item for item, _ in recs]

        hits = len(set(rec_ids) & hidden_items)
        precisions.append(hits / k)
        recalls.append(hits / len(hidden_items))

        train_cats = {cat_map[i] for i in user_train_items.get(user, set()) if i in cat_map}
        rec_cats = [cat_map.get(i) for i in rec_ids]
        in_taste = sum(1 for c in rec_cats if c in train_cats)
        cat_rels.append(in_taste / len(rec_ids))
        evaluated += 1

    metrics = {
        f"precision_at_{k}": round(sum(precisions) / max(evaluated, 1), 4),
        f"recall_at_{k}": round(sum(recalls) / max(evaluated, 1), 4),
        f"category_relevance_at_{k}": round(sum(cat_rels) / max(evaluated, 1), 4),
        "users_evaluated": evaluated,
    }

    out = Path(ev["metrics_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    run_id = Path("models/run_id.txt").read_text().strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({k_: v for k_, v in metrics.items() if isinstance(v, (int, float))})
    print(f"Metrics attached to MLflow run {run_id}")


if __name__ == "__main__":
    main()