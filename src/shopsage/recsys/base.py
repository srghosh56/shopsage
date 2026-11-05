"""Common recommender interface: every model scores all items for a batch of users."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np

from shopsage.recsys.data import Interactions


class Recommender(ABC):
    name: str = "base"
    data: Interactions

    @abstractmethod
    def fit(self, data: Interactions) -> Self: ...

    @abstractmethod
    def score(self, user_idx: np.ndarray) -> np.ndarray:
        """Dense (len(user_idx), n_items) scores; higher is better."""

    def params(self) -> dict[str, Any]:
        return {}

    def recommend(
        self, user_idx: np.ndarray, k: int = 10, exclude_seen: bool = False, batch: int = 1024
    ) -> np.ndarray:
        """Top-k item indices per user, best first."""
        out = np.empty((len(user_idx), k), dtype=np.int64)
        for start in range(0, len(user_idx), batch):
            ub = user_idx[start : start + batch]
            s = self.score(ub).astype(np.float64, copy=True)
            if exclude_seen:
                seen = self.data.binary[ub]
                s[seen.nonzero()] = -np.inf
            # tiny popularity tie-break so equal scores are ordered deterministically
            s += 1e-9 * self.data.item_buyers / max(1.0, self.data.item_buyers.max())
            top = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
            order = np.take_along_axis(s, top, axis=1).argsort(axis=1)[:, ::-1]
            out[start : start + len(ub)] = np.take_along_axis(top, order, axis=1)
        return out

    def recommend_ids(
        self, customer_ids: list[int], k: int = 10, exclude_seen: bool = False
    ) -> dict[int, list[str]]:
        """Stock codes per known customer id (unknown customers are skipped)."""
        known = [c for c in customer_ids if c in self.data.user_index]
        if not known:
            return {}
        idx = np.array([self.data.user_index[c] for c in known])
        top = self.recommend(idx, k, exclude_seen)
        return {
            c: [str(self.data.item_ids[i]) for i in row] for c, row in zip(known, top, strict=True)
        }
