import numpy as np
import pandas as pd
import pytest

from shopsage import config
from shopsage.eval.runner import evaluate_model
from shopsage.eval.split import ground_truth, make_test_split, make_validation_split
from shopsage.recsys.baselines import BuyAgain, Popularity, RecentPopularity
from shopsage.recsys.data import Interactions, load_transactions


@pytest.fixture(scope="module")
def tx(cleaned):
    return load_transactions(config.PROCESSED_DIR)


@pytest.fixture(scope="module")
def train_test(tx):
    tr, te = make_test_split(tx)
    return Interactions.from_lines(tr, end=pd.Timestamp(config.TEST_START)), te


def test_split_has_no_leakage(tx):
    tr, te = make_test_split(tx)
    assert tr.invoice_date.max() < pd.Timestamp(config.TEST_START)
    assert te.invoice_date.min() >= pd.Timestamp(config.TEST_START)
    assert te.invoice_date.max() < pd.Timestamp(config.TEST_END)
    inner, val = make_validation_split(tx)
    assert inner.invoice_date.max() < pd.Timestamp(config.VAL_START) <= val.invoice_date.min()
    assert val.invoice_date.max() < pd.Timestamp(config.TEST_START)


def test_interaction_counts_are_distinct_orders(tx):
    tr, _ = make_test_split(tx)
    data = Interactions.from_lines(tr)
    u, i = 0, data.counts[0].indices[0]
    cid, code = data.user_ids[u], data.item_ids[i]
    expected = tr[(tr.customer_id == cid) & (tr.stock_code == code)].invoice.nunique()
    assert data.counts[u, i] == expected


def test_ground_truth_new_items_excludes_seen(train_test):
    data, te = train_test
    allt = ground_truth(te, data)
    newt = ground_truth(te, data, new_items_only=True)
    assert set(newt) <= set(allt)
    for u, items in newt.items():
        seen = set(data.item_ids[data.binary[data.user_index[u]].indices])
        assert not items & seen


def test_item_cosine_is_symmetric_and_bounded(train_test):
    data, _ = train_test
    a, b = data.item_ids[:2]
    assert data.item_cosine(a, b) == pytest.approx(data.item_cosine(b, a))
    assert data.item_cosine(a, a) == pytest.approx(1.0, abs=1e-5)
    assert data.item_cosine(a, "nope") == 0.0


def test_popularity_ranks_most_bought_first(train_test):
    data, _ = train_test
    top = Popularity().fit(data).recommend(np.array([0, 1]), k=3)
    assert top[0, 0] == int(np.argmax(data.item_buyers))
    assert (top[0] == top[1]).all()


def test_buy_again_puts_bought_items_first(train_test):
    data, _ = train_test
    model = BuyAgain(30).fit(data)
    u = int(np.argmax(np.diff(data.binary.indptr)))  # the user with most distinct items
    n_seen = data.binary[u].nnz
    top = model.recommend(np.array([u]), k=min(10, n_seen))[0]
    assert set(top) <= set(data.binary[u].indices)


def test_exclude_seen_removes_bought_items(train_test):
    data, _ = train_test
    u = 0
    top = RecentPopularity(30).fit(data).recommend(np.array([u]), k=10, exclude_seen=True)[0]
    assert not set(top) & set(data.binary[u].indices)


@pytest.mark.parametrize("protocol", ["all", "new"])
def test_evaluate_model_returns_all_metrics(train_test, protocol):
    data, te = train_test
    res = evaluate_model(BuyAgain().fit(data), data, te, k=5, protocol=protocol)
    for key in [
        "precision@5",
        "recall@5",
        "ndcg@5",
        "map@5",
        "hit_rate@5",
        "coverage@5",
        "diversity@5",
        "pop_bias@5",
    ]:
        assert 0.0 <= res[key] <= 1.0, key
    assert res["n_users"] > 0
