"""
RecOps - Item-based collaborative filtering library.

Shared model logic used by both the training stage and the evaluation stage.
The model is deliberately simple and fast: build a sparse user-item strength
matrix, find each item's top-k most similar items by cosine similarity of
their user-interaction vectors, and recommend by propagating a user's past
item strengths through those similarity lists. Fast retraining (seconds) is a
feature here - it is what makes the live drift->retrain demo snappy.
"""

import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def build_model(interactions: pd.DataFrame, top_k_neighbors: int, min_item_interactions: int) -> dict:
    """Train an item-based CF model from a (user_id, product_id, strength) table."""
    # Drop items too rare to have a meaningful similarity signal.
    item_counts = interactions.groupby("product_id")["user_id"].nunique()
    keep_items = item_counts[item_counts >= min_item_interactions].index
    data = interactions[interactions["product_id"].isin(keep_items)].copy()

    users = data["user_id"].astype("category")
    items = data["product_id"].astype("category")

    matrix = csr_matrix(
        (data["strength"].values, (users.cat.codes, items.cat.codes)),
        shape=(len(users.cat.categories), len(items.cat.categories)),
    )

    # Item vectors = columns; nearest neighbours by cosine among item vectors.
    item_vectors = matrix.T.tocsr()
    n_items = item_vectors.shape[0]
    k = min(top_k_neighbors + 1, n_items)  # +1 because each item's nearest neighbour is itself

    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(item_vectors)
    distances, indices = nn.kneighbors(item_vectors, n_neighbors=k)

    item_ids = items.cat.categories.to_numpy()
    neighbors = {}
    for row in range(n_items):
        sims = []
        for dist, idx in zip(distances[row], indices[row]):
            if idx == row:
                continue  # skip self
            sims.append((int(item_ids[idx]), float(1.0 - dist)))
        neighbors[int(item_ids[row])] = sims[: top_k_neighbors]

    # User profiles: strengths of every item the user has interacted with.
    profiles = {
        user: dict(zip(group["product_id"].astype(int), group["strength"]))
        for user, group in data.groupby("user_id")
    }

    return {
        "neighbors": neighbors,
        "profiles": profiles,
        "n_users": int(matrix.shape[0]),
        "n_items": int(matrix.shape[1]),
    }


def recommend_for_user(model: dict, user_id: str, n: int = 10, exclude_seen: bool = True):
    """Score candidate items by sum(strength(seen_item) * similarity(seen_item, candidate))."""
    profile = model["profiles"].get(user_id)
    if not profile:
        return []  # unknown / cold-start user

    scores = {}
    for seen_item, strength in profile.items():
        for neighbor_item, sim in model["neighbors"].get(seen_item, []):
            if exclude_seen and neighbor_item in profile:
                continue
            scores[neighbor_item] = scores.get(neighbor_item, 0.0) + strength * sim

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(int(item), float(score)) for item, score in ranked[:n]]


def save_model(model: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)