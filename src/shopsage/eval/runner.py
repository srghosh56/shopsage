"""Evaluate any Recommender on a holdout window under two protocols.

- `all`: every item bought in the window is relevant (repeat purchases count).
- `new`: only items the customer had never bought before; already-bought items are
  excluded from the recommendations too. This isolates discovery.
"""

from __future__ import annotations

import pandas as pd

from shopsage.eval.metrics import evaluate_lists
from shopsage.eval.split import ground_truth
from shopsage.recsys.base import Recommender
from shopsage.recsys.data import Interactions

PROTOCOLS = ("all", "new")


def evaluate_model(
    model: Recommender,
    train: Interactions,
    holdout: pd.DataFrame,
    k: int = 10,
    protocol: str = "all",
    beyond_accuracy: bool = True,
) -> dict[str, float]:
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    new_only = protocol == "new"
    truth = ground_truth(holdout, train, new_items_only=new_only)
    recs = model.recommend_ids(list(truth), k=k, exclude_seen=new_only)
    return evaluate_lists(
        recs,
        truth,
        k=k,
        catalog_size=train.n_items,
        popularity_pct=train.popularity_percentile if beyond_accuracy else None,
        similarity=train.item_cosine if beyond_accuracy else None,
    )
