"""Time-based split: train strictly before the cut-off, evaluate on the following window."""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

from shopsage.config import TEST_END, TEST_START, VAL_START
from shopsage.recsys.data import Interactions


def split_by_time(tx: pd.DataFrame, cut: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lines before `cut` and lines in [cut, end)."""
    ts = tx.invoice_date
    cut_ts, end_ts = pd.Timestamp(cut), pd.Timestamp(end)
    return tx[ts < cut_ts].copy(), tx[(ts >= cut_ts) & (ts < end_ts)].copy()


def make_validation_split(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train-inner (< 2011-06-01) vs validation (2011-06-01 .. 2011-09-01): the end of train."""
    train, _ = split_by_time(tx, TEST_START, TEST_END)
    return split_by_time(train, VAL_START, TEST_START)


def make_test_split(tx: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full train (< 2011-09-01) vs test (2011-09-01 .. 2011-12-01)."""
    return split_by_time(tx, TEST_START, TEST_END)


def ground_truth(
    holdout: pd.DataFrame, train: Interactions, new_items_only: bool = False
) -> dict[int, set[str]]:
    """Items each warm user (seen in train) bought in the holdout window.

    With `new_items_only`, items the user already bought in train are dropped, and users left
    with nothing are dropped. Items unknown to train still count (they lower recall honestly).
    """
    warm = holdout[holdout.customer_id.isin(train.user_index)]
    truth: dict[Any, set[Any]] = warm.groupby("customer_id").stock_code.agg(set).to_dict()
    if new_items_only:
        out = {}
        for u, items in truth.items():
            row = train.binary[train.user_index[u]]
            seen = set(train.item_ids[row.indices].astype(str))
            new = items - seen
            if new:
                out[u] = new
        truth = out
    return {int(u): {str(i) for i in s} for u, s in truth.items()}
