"""Tests run on a small fixture (tests/fixtures/raw_sample.parquet), never on the real data."""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="shopsage-tests-"))
os.environ["SHOPSAGE_PROCESSED_DIR"] = str(_TMP / "processed")
os.environ["SHOPSAGE_ARTIFACTS_DIR"] = str(_TMP / "artifacts")
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{_TMP / 'mlflow.db'}"

import pytest  # noqa: E402

from shopsage import config  # noqa: E402
from shopsage.data.clean import Rule, clean  # noqa: E402
from shopsage.data.sample import FIXTURE_RAW  # noqa: E402


@pytest.fixture(scope="session")
def cleaned() -> list[Rule]:
    """Clean the fixture once per test session into a temp directory."""
    return clean(FIXTURE_RAW, config.PROCESSED_DIR, config.DUCKDB_PATH)


@pytest.fixture(scope="session")
def db(cleaned):
    """In-memory views over the cleaned Parquet files (no lock on the DuckDB file)."""
    import duckdb

    from shopsage.data.clean import TABLES

    con = duckdb.connect()
    for t in TABLES:
        con.execute(f"CREATE VIEW {t} AS SELECT * FROM '{config.PROCESSED_DIR / t}.parquet'")
    yield con
    con.close()


@pytest.fixture(scope="session")
def artifacts(cleaned):
    """Serving artifacts fitted on the fixture (small params, no tuning)."""
    import json

    import pandas as pd

    from shopsage.analytics import clv, segments
    from shopsage.recsys.data import load_transactions
    from shopsage.recsys.train import fit_for_serving

    tx = load_transactions(config.PROCESSED_DIR)
    best = {
        "new": {
            "item_knn": {"params": {"n_neighbors": 10, "shrink": 10, "weighting": "log"}},
            "als": {"params": {"factors": 8, "regularization": 0.1, "alpha": 1.0, "iterations": 5}},
        },
        "all": {"buy_again": {"params": {"half_life_days": 90}}},
    }
    fit_for_serving(tx, best)
    full = pd.read_parquet(config.PROCESSED_DIR / "transactions.parquet")
    customers, summary, _ = segments.segment(segments.rfm_table(full), k=4)
    segments.save(customers, summary, config.DUCKDB_PATH)
    clv.serve_clv(clv.orders_table(full), config.DUCKDB_PATH)
    (config.ARTIFACTS_DIR / "test_results.json").write_text(json.dumps({"all": [], "new": []}))
    (config.ARTIFACTS_DIR / "clv_validation.json").write_text(json.dumps({"orders": {}}))
    return config.ARTIFACTS_DIR


@pytest.fixture(scope="session")
def service(artifacts):
    from shopsage.service import ShopSageService

    return ShopSageService(artifacts, config.DUCKDB_PATH)
