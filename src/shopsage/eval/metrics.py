"""Ranking metrics, written by hand (no library) and unit-tested in tests/test_metrics.py.

Per-user metrics take `recs` (ranked list) and `relevant` (set). Relevance is binary.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

Item = str


def precision_at_k(recs: Sequence[Item], relevant: set[Item], k: int) -> float:
    """Share of the top-k slots that are relevant (denominator is always k)."""
    if k <= 0:
        raise ValueError("k must be positive")
    return sum(1 for r in recs[:k] if r in relevant) / k


def recall_at_k(recs: Sequence[Item], relevant: set[Item], k: int) -> float:
    """Share of the relevant items that appear in the top k."""
    if not relevant:
        return 0.0
    return sum(1 for r in recs[:k] if r in relevant) / len(relevant)


def hit_rate_at_k(recs: Sequence[Item], relevant: set[Item], k: int) -> float:
    """1 if at least one relevant item is in the top k, else 0."""
    return float(any(r in relevant for r in recs[:k]))


def dcg_at_k(recs: Sequence[Item], relevant: set[Item], k: int) -> float:
    return sum(1.0 / math.log2(i + 2) for i, r in enumerate(recs[:k]) if r in relevant)


def ndcg_at_k(recs: Sequence[Item], relevant: set[Item], k: int) -> float:
    """DCG of the list divided by the DCG of a perfect list (all relevant items first)."""
    if not relevant:
        return 0.0
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg_at_k(recs, relevant, k) / ideal


def average_precision_at_k(recs: Sequence[Item], relevant: set[Item], k: int) -> float:
    """Mean of precision@i at each rank i that holds a relevant item, divided by min(|rel|, k)."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, r in enumerate(recs[:k]):
        if r in relevant:
            hits += 1
            total += hits / (i + 1)
    return total / min(len(relevant), k)


def catalog_coverage(all_recs: Iterable[Sequence[Item]], catalog_size: int, k: int) -> float:
    """Share of the catalogue that appears in at least one user's top k."""
    shown: set[Item] = set()
    for recs in all_recs:
        shown.update(recs[:k])
    return len(shown) / catalog_size if catalog_size else 0.0


def intra_list_diversity(
    recs: Sequence[Item], similarity: Callable[[Item, Item], float], k: int
) -> float:
    """1 - mean pairwise similarity of the items in the top k (0 = all identical)."""
    top = list(recs[:k])
    if len(top) < 2:
        return 0.0
    sims = [similarity(a, b) for i, a in enumerate(top) for b in top[i + 1 :]]
    return 1.0 - float(np.mean(sims))


def popularity_bias(recs: Sequence[Item], popularity_pct: Mapping[Item, float], k: int) -> float:
    """Mean popularity percentile (0..1, 1 = best-seller) of the recommended items."""
    top = recs[:k]
    if not top:
        return 0.0
    return float(np.mean([popularity_pct.get(r, 0.0) for r in top]))


ACCURACY_METRICS: dict[str, Callable[[Sequence[Item], set[Item], int], float]] = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "hit_rate": hit_rate_at_k,
}


def evaluate_lists(
    recs: Mapping[Any, Sequence[Item]],
    truth: Mapping[Any, set[Item]],
    k: int,
    catalog_size: int,
    popularity_pct: Mapping[Item, float] | None = None,
    similarity: Callable[[Item, Item], float] | None = None,
) -> dict[str, float]:
    """Average every metric over the users in `truth` (users without recs score 0)."""
    users = [u for u in truth if truth[u]]
    out: dict[str, float] = {}
    for name, fn in ACCURACY_METRICS.items():
        out[f"{name}@{k}"] = float(np.mean([fn(recs.get(u, []), truth[u], k) for u in users]))
    out[f"coverage@{k}"] = catalog_coverage((recs.get(u, []) for u in users), catalog_size, k)
    if similarity is not None:
        out[f"diversity@{k}"] = float(
            np.mean([intra_list_diversity(recs.get(u, []), similarity, k) for u in users])
        )
    if popularity_pct is not None:
        out[f"pop_bias@{k}"] = float(
            np.mean([popularity_bias(recs.get(u, []), popularity_pct, k) for u in users])
        )
    out["n_users"] = float(len(users))
    return out
