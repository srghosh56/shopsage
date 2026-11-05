import numpy as np
import pandas as pd
import pytest

from shopsage import config
from shopsage.analytics import clv, segments


@pytest.fixture(scope="module")
def tx(cleaned):
    return pd.read_parquet(config.PROCESSED_DIR / "transactions.parquet")


def test_rfm_table(tx):
    rfm = segments.rfm_table(tx)
    assert len(rfm) == tx.customer_id.nunique()
    assert (rfm.recency >= 0).all()  # whole days; 0 = bought on the last day
    assert (rfm.frequency >= 1).all() and (rfm.monetary > 0).all()
    c = rfm.iloc[0]
    sub = tx[tx.customer_id == c.customer_id]
    assert c.frequency == sub.invoice.nunique()
    assert c.monetary == pytest.approx(sub.revenue.sum(), abs=0.01)


@pytest.mark.parametrize(
    ("r", "f", "m", "name"),
    [
        (-1, 1.5, 1.5, "Champions"),
        (-1, 0.5, 0.5, "Loyal regulars"),
        (-1, -1, -1, "Promising new"),
        (1, 1, 1, "At risk"),
        (1, -1, -1, "Hibernating"),
        (0.2, -1, -1, "Needs attention"),
    ],
)
def test_name_segment(r, f, m, name):
    assert segments.name_segment(r, f, m) == name


def test_segment_assigns_everyone(tx):
    customers, summary, diag = segments.segment(segments.rfm_table(tx), k=4)
    assert customers.segment.notna().all()
    assert diag["k"] == 4
    assert summary.customers.sum() == len(customers)
    assert summary.share_revenue.sum() == pytest.approx(1.0)
    assert summary.action.notna().all()


def test_clv_error_metrics_and_top_share():
    pred, actual = np.array([1.0, 2.0, 3.0]), np.array([1.0, 3.0, 3.0])
    m = clv.error_metrics(pred, actual)
    assert m["MAE"] == pytest.approx(1 / 3)
    assert m["total_error_pct"] == pytest.approx(-1 / 7 * 100)
    assert (
        clv.top_share(np.array([3.0, 2.0, 1.0, 0.0, 0.0]), np.array([10.0, 0, 0, 0, 10.0])) == 0.5
    )


def test_clv_fit_and_predict_on_fixture(tx):
    from pymc_marketing.clv.utils import rfm_summary

    orders = clv.orders_table(tx)
    s = rfm_summary(orders, "customer_id", "date", monetary_value_col="value")
    bg, gg = clv.fit_models(s)
    pred = clv.predict(bg, gg, s, days=90)
    assert len(pred) == len(s)
    assert (pred.expected_orders >= 0).all()
    assert pred.p_alive.between(0, 1).all()
    assert (pred.expected_order_value > 0).all()
