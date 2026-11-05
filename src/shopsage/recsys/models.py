"""Collaborative-filtering models: item-kNN (written here) and ALS (the `implicit` library)."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import scipy.sparse as sp
from implicit.cpu.als import AlternatingLeastSquares

from shopsage.config import SEED
from shopsage.recsys.base import Recommender
from shopsage.recsys.data import Interactions


def _weighted(counts: sp.csr_matrix, weighting: str) -> sp.csr_matrix:
    """User x item weights: binary, log(1 + #orders) or raw #orders."""
    m = counts.astype(np.float32).tocsr(copy=True)
    if weighting == "binary":
        m.data[:] = 1.0
    elif weighting == "log":
        m.data = np.log1p(m.data)
    elif weighting != "count":
        raise ValueError(weighting)
    return m


class ItemKNN(Recommender):
    """Item-based collaborative filtering.

    sim(i, j) = co-buyers(i, j) / (sqrt(buyers(i) * buyers(j)) + shrink), keeping each item's
    `n_neighbors` most similar items. A user's score for item j sums sim(i, j) over the items i
    they bought, weighted by how often they bought i.
    """

    name = "item_knn"

    def __init__(
        self, n_neighbors: int = 100, shrink: float = 10.0, weighting: str = "log"
    ) -> None:
        self.n_neighbors = n_neighbors
        self.shrink = shrink
        self.weighting = weighting

    def params(self) -> dict[str, Any]:
        return {"n_neighbors": self.n_neighbors, "shrink": self.shrink, "weighting": self.weighting}

    def fit(self, data: Interactions) -> Self:
        self.data = data
        self._user_items = _weighted(data.counts, self.weighting)
        b = data.binary.astype(np.float32)
        co = (b.T @ b).toarray()  # co-buyer counts
        np.fill_diagonal(co, 0.0)
        n = data.item_buyers.astype(np.float32)
        sim = co / (np.sqrt(np.outer(n, n)) + self.shrink)
        if self.n_neighbors < sim.shape[1]:
            # keep the top-n neighbours per row (item i -> its most similar items j)
            drop = np.argpartition(-sim, self.n_neighbors, axis=1)[:, self.n_neighbors :]
            np.put_along_axis(sim, drop, 0.0, axis=1)
        self.similarity = sp.csr_matrix(sim)
        self.co_counts = co.astype(np.int32)
        return self

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        return np.asarray((self._user_items[user_idx] @ self.similarity).toarray())


class ALS(Recommender):
    """Implicit-feedback matrix factorisation (Hu, Koren & Volinsky 2008) via `implicit`.

    Confidence = 1 + alpha * weight(#orders). Score = user_factors . item_factors.
    """

    name = "als"

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 10.0,
        iterations: int = 20,
        weighting: str = "log",
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.weighting = weighting

    def params(self) -> dict[str, Any]:
        return {
            "factors": self.factors,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "iterations": self.iterations,
            "weighting": self.weighting,
        }

    def fit(self, data: Interactions) -> Self:
        self.data = data
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=SEED,
            num_threads=1,  # deterministic results
        )
        self.model.fit(_weighted(data.counts, self.weighting), show_progress=False)
        return self

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        uf = np.asarray(self.model.user_factors)[user_idx]
        return uf @ np.asarray(self.model.item_factors).T
