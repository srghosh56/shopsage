"""Customer x item interaction matrices built from the clean transactions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from shopsage.config import PROCESSED_DIR


def load_transactions(processed_dir: Path | None = None) -> pd.DataFrame:
    path = (processed_dir or PROCESSED_DIR) / "transactions.parquet"
    df = pd.read_parquet(path, columns=["invoice", "stock_code", "invoice_date", "customer_id"])
    df["customer_id"] = df["customer_id"].astype("int64")
    return df


@dataclass
class Interactions:
    """Who bought what before `end`. `counts[u, i]` = number of distinct orders containing i."""

    user_ids: np.ndarray
    item_ids: np.ndarray
    counts: sp.csr_matrix
    lines: pd.DataFrame = field(repr=False)
    end: pd.Timestamp

    @classmethod
    def from_lines(cls, lines: pd.DataFrame, end: pd.Timestamp | None = None) -> Interactions:
        pairs = lines.groupby(["customer_id", "stock_code"]).invoice.nunique().reset_index()
        user_ids = np.sort(pairs.customer_id.unique())
        item_ids = np.sort(pairs.stock_code.unique())
        rows = np.searchsorted(user_ids, pairs.customer_id.to_numpy())
        cols = np.searchsorted(item_ids, pairs.stock_code.to_numpy())
        counts = sp.csr_matrix(
            (pairs.invoice.to_numpy(dtype=np.float32), (rows, cols)),
            shape=(len(user_ids), len(item_ids)),
        )
        end = end if end is not None else lines.invoice_date.max()
        return cls(user_ids, item_ids, counts, lines, pd.Timestamp(end))

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @cached_property
    def user_index(self) -> dict[int, int]:
        return {int(u): i for i, u in enumerate(self.user_ids)}

    @cached_property
    def item_index(self) -> dict[str, int]:
        return {str(s): i for i, s in enumerate(self.item_ids)}

    @cached_property
    def binary(self) -> sp.csr_matrix:
        b = self.counts.copy()
        b.data[:] = 1.0
        return b

    @cached_property
    def item_buyers(self) -> np.ndarray:
        return np.asarray(self.binary.sum(axis=0)).ravel()

    @cached_property
    def popularity_percentile(self) -> dict[str, float]:
        """Percentile rank of each item by number of buyers (1.0 = most bought)."""
        pct = pd.Series(self.item_buyers).rank(pct=True).to_numpy()
        return dict(zip(self.item_ids.astype(str), pct, strict=True))

    @cached_property
    def item_item_cosine(self) -> np.ndarray:
        """Dense item x item cosine similarity of buyer vectors (float32)."""
        cols = self.binary.T.tocsr().astype(np.float32)
        norms = np.sqrt(np.asarray(cols.multiply(cols).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        unit = sp.csr_matrix(sp.diags(1.0 / norms) @ cols)
        return np.asarray((unit @ unit.T).toarray(), dtype=np.float32)

    def item_cosine(self, a: str, b: str) -> float:
        """Cosine similarity between two items' buyer vectors."""
        ia, ib = self.item_index.get(a), self.item_index.get(b)
        if ia is None or ib is None:
            return 0.0
        return float(self.item_item_cosine[ia, ib])

    def similarity_fn(self) -> Callable[[str, str], float]:
        return self.item_cosine
