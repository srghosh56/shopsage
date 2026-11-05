"""Non-personalised and simple personalised baselines."""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import pandas as pd
import scipy.sparse as sp

from shopsage.recsys.base import Recommender
from shopsage.recsys.data import Interactions


class Popularity(Recommender):
    """Items ranked by number of distinct buyers over the whole training period."""

    name = "popularity"

    def fit(self, data: Interactions) -> Self:
        self.data = data
        self._scores = data.item_buyers.astype(np.float64)
        return self

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        return np.tile(self._scores, (len(user_idx), 1))


class RecentPopularity(Popularity):
    """Items ranked by number of distinct buyers in the last `window_days` of training."""

    name = "recent_popularity"

    def __init__(self, window_days: int = 30) -> None:
        self.window_days = window_days

    def params(self) -> dict[str, Any]:
        return {"window_days": self.window_days}

    def fit(self, data: Interactions) -> Self:
        self.data = data
        start = data.end - pd.Timedelta(days=self.window_days)
        recent = data.lines[data.lines.invoice_date > start]
        buyers = recent.groupby("stock_code").customer_id.nunique()
        s = np.zeros(data.n_items)
        for code, n in buyers.items():
            i = data.item_index.get(str(code))
            if i is not None:
                s[i] = n
        # fall back to all-time popularity for ordering items with no recent sales
        self._scores = s + data.item_buyers / (data.item_buyers.max() + 1.0)
        return self


class BuyAgain(Recommender):
    """Re-recommend what the customer bought before, weighted by recency-decayed frequency.

    Each past order containing the item adds exp(-age_days / half_life * ln 2). Items the user
    never bought are filled in by all-time popularity, always ranked below bought items.
    """

    name = "buy_again"

    def __init__(self, half_life_days: float = 90.0) -> None:
        self.half_life_days = half_life_days

    def params(self) -> dict[str, Any]:
        return {"half_life_days": self.half_life_days}

    def fit(self, data: Interactions) -> Self:
        self.data = data
        orders = (
            data.lines.groupby(["customer_id", "stock_code", "invoice"])
            .invoice_date.min()
            .reset_index()
        )
        age = (data.end - orders.invoice_date).dt.total_seconds() / 86400.0
        w = np.exp(-np.log(2) * age.to_numpy() / self.half_life_days)
        rows = orders.customer_id.map(data.user_index).to_numpy()
        cols = orders.stock_code.map(data.item_index).to_numpy()
        self._decayed = sp.csr_matrix((w, (rows, cols)), shape=data.counts.shape)
        self._pop = data.item_buyers / (data.item_buyers.max() + 1.0)
        return self

    def score(self, user_idx: np.ndarray) -> np.ndarray:
        d = self._decayed[user_idx].toarray()
        bought = d > 0
        return np.where(bought, 1.0 + d, 0.0) + np.where(bought, 0.0, self._pop)
