import json

import numpy as np
import pandas as pd
import pytest

from shopsage import config
from shopsage.eval.split import make_test_split
from shopsage.recsys.data import Interactions, load_transactions
from shopsage.recsys.models import ALS, ItemKNN, _weighted
from shopsage.recsys.train import grid, tune


@pytest.fixture(scope="module")
def data(cleaned):
    tr, _ = make_test_split(load_transactions(config.PROCESSED_DIR))
    return Interactions.from_lines(tr, end=pd.Timestamp(config.TEST_START))


def test_weighting():
    import scipy.sparse as sp

    m = sp.csr_matrix(np.array([[0, 3], [1, 0]], dtype=np.float32))
    assert _weighted(m, "binary").toarray().tolist() == [[0, 1], [1, 0]]
    assert _weighted(m, "log")[0, 1] == pytest.approx(np.log1p(3))
    with pytest.raises(ValueError):
        _weighted(m, "bad")


def test_item_knn_similarity(data):
    model = ItemKNN(n_neighbors=5, shrink=0, weighting="binary").fit(data)
    sim = model.similarity
    assert sim.diagonal().max() == 0  # no self-similarity
    assert (np.diff(sim.indptr) <= 5).all()  # at most n_neighbours per item
    assert sim.max() <= 1.0 + 1e-6
    # co-purchase counts are symmetric
    assert (model.co_counts == model.co_counts.T).all()


def test_item_knn_scores_shape(data):
    model = ItemKNN(n_neighbors=20).fit(data)
    s = model.score(np.array([0, 1, 2]))
    assert s.shape == (3, data.n_items)


def test_als_is_deterministic(data):
    a = ALS(factors=8, iterations=3).fit(data).score(np.array([0]))
    b = ALS(factors=8, iterations=3).fit(data).score(np.array([0]))
    assert np.allclose(a, b)


def test_tune_logs_and_returns_best(cleaned, tmp_path):
    tx = load_transactions(config.PROCESSED_DIR)
    space = {"item_knn": (ItemKNN, grid(n_neighbors=[5, 10], shrink=[0], weighting=["binary"]))}
    best = tune(tx, space=space)
    assert set(best) == {"all", "new"}
    assert best["all"]["item_knn"]["params"]["n_neighbors"] in (5, 10)
    json.dumps(best, default=str)
