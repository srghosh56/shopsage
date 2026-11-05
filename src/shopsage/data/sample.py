"""Build a small raw-data fixture so tests run without the full download.

All rows of a seeded random set of customers (most of them active both before and after the
test cut-off), plus some guest-checkout and non-product rows so every cleaning rule fires.
The data is UCI Online Retail II (CC BY 4.0, Chen 2019).
"""

from __future__ import annotations

import duckdb

from shopsage.config import FIXTURE_DIR, SEED, TEST_START
from shopsage.data.download import RAW_PARQUET

FIXTURE_RAW = FIXTURE_DIR / "raw_sample.parquet"


def build_fixture(n_active: int = 100, n_other: int = 30) -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SELECT setseed({SEED / 100})")
    con.execute(f"CREATE TABLE r AS SELECT * FROM read_parquet('{RAW_PARQUET}')")
    con.execute(
        f"""
        CREATE TABLE picked AS
        WITH spans AS (
            SELECT "Customer ID" AS cid,
                   min(InvoiceDate) < DATE '{TEST_START}' AND max(InvoiceDate) >= DATE '{TEST_START}'
                       AS both_sides
            FROM r WHERE "Customer ID" IS NOT NULL GROUP BY 1 ORDER BY 1
        )
        (SELECT cid FROM spans WHERE both_sides ORDER BY random() LIMIT {n_active})
        UNION ALL
        (SELECT cid FROM spans WHERE NOT both_sides ORDER BY random() LIMIT {n_other})
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT * FROM r WHERE "Customer ID" IN (SELECT cid FROM picked)
            UNION ALL (SELECT * FROM r WHERE "Customer ID" IS NULL ORDER BY random() LIMIT 500)
            UNION ALL (SELECT * FROM r WHERE NOT regexp_matches(StockCode, '^[0-9]{{5}}')
                       ORDER BY random() LIMIT 50)
            UNION ALL (SELECT * FROM r WHERE Invoice LIKE 'A%')
        ) TO '{FIXTURE_RAW}' (FORMAT parquet, COMPRESSION zstd)
        """
    )


if __name__ == "__main__":
    build_fixture()
    print(FIXTURE_RAW, FIXTURE_RAW.stat().st_size)
