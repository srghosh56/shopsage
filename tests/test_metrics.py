import math

import pytest

from shopsage.eval import metrics as m

RECS = ["a", "b", "c", "d", "e"]
REL = {"a", "c", "z"}  # z is never recommended


def test_precision_at_k():
    assert m.precision_at_k(RECS, REL, 1) == 1.0
    assert m.precision_at_k(RECS, REL, 3) == pytest.approx(2 / 3)
    assert m.precision_at_k(RECS, REL, 5) == pytest.approx(2 / 5)
    # short list still divides by k
    assert m.precision_at_k(["a"], REL, 5) == pytest.approx(1 / 5)
    with pytest.raises(ValueError):
        m.precision_at_k(RECS, REL, 0)


def test_recall_at_k():
    assert m.recall_at_k(RECS, REL, 1) == pytest.approx(1 / 3)
    assert m.recall_at_k(RECS, REL, 5) == pytest.approx(2 / 3)
    assert m.recall_at_k(RECS, set(), 5) == 0.0


def test_hit_rate():
    assert m.hit_rate_at_k(RECS, REL, 1) == 1.0
    assert m.hit_rate_at_k(["b", "d"], REL, 2) == 0.0


def test_ndcg_hand_computed():
    # hits at ranks 1 and 3 -> DCG = 1/log2(2) + 1/log2(4) = 1.5
    # ideal with 3 relevant, k=5 -> 1 + 1/log2(3) + 1/log2(4)
    ideal = 1 + 1 / math.log2(3) + 0.5
    assert m.ndcg_at_k(RECS, REL, 5) == pytest.approx(1.5 / ideal)
    assert m.ndcg_at_k(["a", "c", "z"], REL, 3) == pytest.approx(1.0)
    assert m.ndcg_at_k(["x", "y"], REL, 2) == 0.0


def test_ndcg_ideal_is_capped_at_k():
    # only one slot: a perfect first item gives NDCG 1 even with 3 relevant items
    assert m.ndcg_at_k(RECS, REL, 1) == pytest.approx(1.0)


def test_average_precision_hand_computed():
    # hits at 1 (P=1) and 3 (P=2/3); divided by min(3, 5) = 3
    assert m.average_precision_at_k(RECS, REL, 5) == pytest.approx((1 + 2 / 3) / 3)
    # k=2: one hit, divided by min(3, 2) = 2
    assert m.average_precision_at_k(RECS, REL, 2) == pytest.approx(0.5)


def test_coverage():
    lists = [["a", "b"], ["b", "c"], ["a", "x"]]
    assert m.catalog_coverage(lists, 10, 2) == pytest.approx(4 / 10)
    assert m.catalog_coverage(lists, 10, 1) == pytest.approx(2 / 10)


def test_intra_list_diversity():
    same = lambda a, b: 1.0  # noqa: E731
    different = lambda a, b: 0.0  # noqa: E731
    assert m.intra_list_diversity(RECS, same, 5) == 0.0
    assert m.intra_list_diversity(RECS, different, 5) == 1.0
    assert m.intra_list_diversity(["a"], different, 5) == 0.0


def test_popularity_bias():
    pct = {"a": 1.0, "b": 0.5}
    assert m.popularity_bias(["a", "b", "q"], pct, 3) == pytest.approx(0.5)


def test_evaluate_lists_averages_over_users_with_truth():
    recs = {1: ["a", "b"], 2: ["x", "y"]}
    truth = {1: {"a"}, 2: {"y"}, 3: set()}  # user 3 has no truth -> ignored
    res = m.evaluate_lists(recs, truth, k=2, catalog_size=4)
    assert res["n_users"] == 2
    assert res["hit_rate@2"] == 1.0
    assert res["precision@2"] == pytest.approx(0.5)
    assert res["ndcg@2"] == pytest.approx((1 + 1 / math.log2(3)) / 2)
    assert res["coverage@2"] == pytest.approx(1.0)


def test_user_without_recs_scores_zero():
    res = m.evaluate_lists({}, {1: {"a"}}, k=5, catalog_size=3)
    assert res["recall@5"] == 0.0
