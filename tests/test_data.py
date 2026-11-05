import pandas as pd
import pytest

from shopsage import config
from shopsage.data.clean import PRODUCT_CODE_REGEX, clean, rules_markdown
from shopsage.data.download import RAW_PARQUET, check_raw


def scalar(db, sql):
    return db.execute(sql).fetchone()[0]


def test_every_rule_fires_on_fixture(cleaned):
    counts = {r.name: r.rows for r in cleaned}
    for name in [
        "exact_duplicates",
        "cancellations_to_returns",
        "non_product_codes",
        "missing_customer_id",
        "final_transactions",
    ]:
        assert counts[name] > 0, name


def test_transactions_are_valid_purchases(db):
    assert scalar(db, "SELECT count(*) FROM transactions WHERE invoice LIKE 'C%'") == 0
    assert scalar(db, "SELECT count(*) FROM transactions WHERE quantity <= 0 OR price <= 0") == 0
    assert scalar(db, "SELECT count(*) FROM transactions WHERE customer_id IS NULL") == 0
    bad = scalar(
        db,
        f"SELECT count(*) FROM transactions WHERE NOT regexp_matches(stock_code, '{PRODUCT_CODE_REGEX}')",
    )
    assert bad == 0


def test_one_description_per_stock_code(db):
    n = scalar(
        db,
        "SELECT count(*) FROM (SELECT stock_code FROM lines_all GROUP BY 1 "
        "HAVING count(DISTINCT description) > 1)",
    )
    assert n == 0


def test_no_exact_duplicates(db):
    total = scalar(db, "SELECT count(*) FROM lines_all")
    distinct = scalar(db, "SELECT count(*) FROM (SELECT DISTINCT * FROM lines_all)")
    assert total == distinct


def test_returns_have_positive_quantity(db):
    assert scalar(db, "SELECT count(*) FROM returns") > 0
    assert scalar(db, "SELECT min(quantity) FROM returns") > 0


def test_revenue_matches_quantity_times_price(db):
    diff = scalar(db, "SELECT max(abs(revenue - quantity * price)) FROM transactions")
    assert diff < 0.01


def test_customers_and_products_consistent(db):
    assert scalar(db, "SELECT count(*) FROM customers") == scalar(
        db, "SELECT count(DISTINCT customer_id) FROM transactions"
    )
    assert scalar(db, "SELECT count(*) FROM products") == scalar(
        db, "SELECT count(DISTINCT stock_code) FROM lines_all"
    )


def test_rules_markdown_has_all_rules(cleaned):
    md = rules_markdown(cleaned)
    assert all(r.name in md for r in cleaned)


def test_check_raw_rejects_small_frames():
    df = pd.DataFrame({c: [] for c in ["Invoice", "StockCode"]})
    with pytest.raises(ValueError):
        check_raw(df)


@pytest.mark.slow
@pytest.mark.skipif(not RAW_PARQUET.exists(), reason="full dataset not downloaded")
def test_full_dataset_counts(tmp_path):
    rules = {r.name: r.rows for r in clean(RAW_PARQUET, tmp_path, None)}
    assert rules["raw"] == 1_067_371
    assert rules["final_transactions"] > 700_000
    assert config.UCI_URL.startswith("https://archive.ics.uci.edu/")
