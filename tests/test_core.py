"""
RecOps - Unit tests for the core pipeline logic.

These tests run fast (seconds), need no MLflow server, no DVC remote and no
network, so they are suitable as the first gate in CI: if the maths of the
recommender, the validation gate or the simulator parsing is broken, the
pipeline must not proceed to retraining or image building.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from simulate_events import parse_mix, PERSONAS
from recommender import build_model, recommend_for_user


# ---------- simulator ----------

def test_parse_mix_normalises_weights():
    mix = parse_mix("sneakerhead:2,ethnic:2")
    assert mix["sneakerhead"] == pytest.approx(0.5)
    assert mix["ethnic"] == pytest.approx(0.5)
    assert sum(mix.values()) == pytest.approx(1.0)


def test_parse_mix_rejects_unknown_persona():
    with pytest.raises(ValueError):
        parse_mix("astronaut:1.0")


def test_personas_reference_plausible_segments():
    for name, spec in PERSONAS.items():
        assert spec["gender"], name
        assert spec["categories"], name


# ---------- recommender ----------

def toy_interactions():
    # Two taste clusters: users a,b,c like items 1,2,3; users x,y,z like 7,8,9.
    rows = []
    for u in ["a", "b", "c"]:
        for i in [1, 2, 3]:
            rows.append((u, i, 3.0))
    for u in ["x", "y", "z"]:
        for i in [7, 8, 9]:
            rows.append((u, i, 3.0))
    # user "a" has not seen item 3? she has - so hide one: drop (a,3)
    rows = [r for r in rows if r != ("a", 3, 3.0)]
    return pd.DataFrame(rows, columns=["user_id", "product_id", "strength"])


def test_cf_recommends_within_taste_cluster():
    model = build_model(toy_interactions(), top_k_neighbors=5, min_item_interactions=2)
    recs = recommend_for_user(model, "a", n=3)
    rec_ids = [i for i, _ in recs]
    assert 3 in rec_ids            # the hidden same-cluster item comes back
    assert not set(rec_ids) & {7, 8, 9}   # never crosses into the other cluster


def test_cf_excludes_already_seen_items():
    model = build_model(toy_interactions(), top_k_neighbors=5, min_item_interactions=2)
    recs = recommend_for_user(model, "a", n=10)
    assert not {1, 2} & {i for i, _ in recs}


def test_cf_cold_start_returns_empty():
    model = build_model(toy_interactions(), top_k_neighbors=5, min_item_interactions=2)
    assert recommend_for_user(model, "stranger", n=5) == []


def test_min_interactions_filter_drops_rare_items():
    df = toy_interactions()
    df = pd.concat([df, pd.DataFrame([("a", 99, 1.0)],
                    columns=df.columns)], ignore_index=True)  # item 99: 1 user only
    model = build_model(df, top_k_neighbors=5, min_item_interactions=2)
    assert 99 not in model["neighbors"]
