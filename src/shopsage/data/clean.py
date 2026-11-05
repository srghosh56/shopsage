"""Clean the raw Online Retail II data with DuckDB and write Parquet + a DuckDB database.

Every rule records how many rows it touched; the counts for the full dataset are written
into docs/DECISIONS.md between the CLEANING-RULES markers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

from shopsage.config import DOCS_DIR, DUCKDB_PATH, PROCESSED_DIR
from shopsage.data.download import RAW_PARQUET

log = logging.getLogger(__name__)

# A product stock code starts with five digits (optionally followed by letters, e.g. 85123A).
# Everything else is postage, fees, manual adjustments, vouchers, tests, samples, etc.
PRODUCT_CODE_REGEX = r"^[0-9]{5}"
TABLES = ("transactions", "lines_all", "returns", "products", "customers")


@dataclass
class Rule:
    name: str
    description: str
    rows: int


def _count(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    assert row is not None
    return int(row[0])


def clean(
    raw_path: Path = RAW_PARQUET,
    out_dir: Path = PROCESSED_DIR,
    db_path: Path | None = DUCKDB_PATH,
) -> list[Rule]:
    """Run all cleaning rules. Returns the rules with affected row counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    rules: list[Rule] = []

    con.execute(
        f"""
        CREATE TABLE raw AS
        SELECT
            trim(CAST(Invoice AS VARCHAR))          AS invoice,
            upper(trim(CAST(StockCode AS VARCHAR))) AS stock_code,
            regexp_replace(upper(trim(Description)), '\\s+', ' ', 'g') AS description,
            CAST(Quantity AS INTEGER)               AS quantity,
            CAST(InvoiceDate AS TIMESTAMP)          AS invoice_date,
            CAST(Price AS DOUBLE)                   AS price,
            CAST("Customer ID" AS INTEGER)          AS customer_id,
            trim(Country)                           AS country
        FROM read_parquet('{raw_path}')
        """
    )
    n_raw = _count(con, "SELECT count(*) FROM raw")
    rules.append(Rule("raw", "Rows in the downloaded file (both Excel sheets).", n_raw))

    # 1. Exact duplicates (the two sheets overlap on 2010-12-01..09, plus repeated lines).
    con.execute("CREATE TABLE d AS SELECT DISTINCT * FROM raw")
    rules.append(
        Rule(
            "exact_duplicates",
            "Identical rows on all eight columns; one copy kept. Mostly the 2010-12-01..09 "
            "overlap between the two Excel sheets.",
            n_raw - _count(con, "SELECT count(*) FROM d"),
        )
    )

    # 2. Cancellations (invoice starts with 'C') are returns, not purchases.
    rules.append(
        Rule(
            "cancellations_to_returns",
            "Invoice starting with 'C' is a cancellation. Moved to the `returns` table "
            "(quantity made positive) and removed from purchases.",
            _count(con, "SELECT count(*) FROM d WHERE invoice LIKE 'C%'"),
        )
    )
    con.execute(
        f"""
        CREATE TABLE returns AS
        SELECT invoice, stock_code, -quantity AS quantity, invoice_date, price,
               customer_id, country
        FROM d
        WHERE invoice LIKE 'C%' AND regexp_matches(stock_code, '{PRODUCT_CODE_REGEX}')
        """
    )

    # 3. Bad-debt adjustment invoices ('A' prefix).
    rules.append(
        Rule(
            "bad_debt_adjustments",
            "Invoice starting with 'A' (accounting 'adjust bad debt' entries). Removed.",
            _count(con, "SELECT count(*) FROM d WHERE invoice LIKE 'A%'"),
        )
    )
    con.execute("CREATE TABLE p AS SELECT * FROM d WHERE regexp_matches(invoice, '^[0-9]+$')")

    # 4. Non-product stock codes.
    rules.append(
        Rule(
            "non_product_codes",
            "Stock code does not start with five digits: POST, DOT, M, C2, D, S, BANK CHARGES, "
            "AMAZONFEE, ADJUST, gift vouchers, TEST, DCGS web-shop codes. Removed.",
            _count(
                con,
                f"SELECT count(*) FROM p WHERE NOT regexp_matches(stock_code, "
                f"'{PRODUCT_CODE_REGEX}')",
            ),
        )
    )
    con.execute(f"DELETE FROM p WHERE NOT regexp_matches(stock_code, '{PRODUCT_CODE_REGEX}')")

    # 5. Non-positive quantity on a non-cancellation invoice (stock write-offs, 'damaged', ...).
    rules.append(
        Rule(
            "non_positive_quantity",
            "Quantity <= 0 on a normal invoice: stock adjustments such as 'damaged', "
            "'lost', 'found'. Removed.",
            _count(con, "SELECT count(*) FROM p WHERE quantity <= 0"),
        )
    )
    con.execute("DELETE FROM p WHERE quantity <= 0")

    # 6. Non-positive price.
    rules.append(
        Rule(
            "non_positive_price",
            "Price <= 0 (free samples, internal moves, negative adjustments). Removed.",
            _count(con, "SELECT count(*) FROM p WHERE price <= 0"),
        )
    )
    con.execute("DELETE FROM p WHERE price <= 0")

    # 7. Purchases that were later cancelled in full by the same customer.
    # For each (customer, stock code, quantity) key with k cancellations, drop the k most
    # recent purchases with that key that happened before the cancellation date.
    con.execute(
        """
        CREATE TABLE matched AS
        WITH c AS (
            SELECT customer_id, stock_code, quantity, count(*) AS k, max(invoice_date) AS last_c
            FROM returns WHERE customer_id IS NOT NULL
            GROUP BY ALL
        ),
        ranked AS (
            SELECT p.rowid AS rid,
                   row_number() OVER (PARTITION BY p.customer_id, p.stock_code, p.quantity
                                      ORDER BY p.invoice_date DESC) AS rk,
                   c.k
            FROM p JOIN c USING (customer_id, stock_code, quantity)
            WHERE p.invoice_date <= c.last_c
        )
        SELECT rid FROM ranked WHERE rk <= k
        """
    )
    rules.append(
        Rule(
            "fully_cancelled_purchases",
            "Purchase line matched one-to-one to a later cancellation by the same customer "
            "with the same stock code and quantity (removes e.g. the 80,995-unit order that "
            "was cancelled the same day). Removed.",
            _count(con, "SELECT count(*) FROM matched"),
        )
    )
    con.execute("DELETE FROM p WHERE rowid IN (SELECT rid FROM matched)")

    # 8. One description per stock code: the most frequent non-null one (ties: alphabetical).
    rules.append(
        Rule(
            "description_conflicts",
            "Stock codes that appear with more than one description (count is stock codes, "
            "not rows). Each gets its most frequent description; ties broken alphabetically.",
            _count(
                con,
                "SELECT count(*) FROM (SELECT stock_code FROM p WHERE description IS NOT NULL "
                "GROUP BY stock_code HAVING count(DISTINCT description) > 1)",
            ),
        )
    )
    con.execute(
        """
        CREATE TABLE desc_map AS
        SELECT stock_code, description FROM (
            SELECT stock_code, description,
                   row_number() OVER (PARTITION BY stock_code
                                      ORDER BY count(*) DESC, description) AS rk
            FROM p WHERE description IS NOT NULL
            GROUP BY stock_code, description
        ) WHERE rk = 1
        """
    )
    con.execute(
        """
        CREATE TABLE lines_all AS
        SELECT p.invoice, p.stock_code, coalesce(m.description, 'UNKNOWN') AS description,
               p.quantity, p.invoice_date, p.price, round(p.quantity * p.price, 2) AS revenue,
               p.customer_id, p.country
        FROM p LEFT JOIN desc_map m USING (stock_code)
        ORDER BY p.invoice_date, p.invoice, p.stock_code
        """
    )

    # 9. Missing customer ID: kept in lines_all (sales EDA) but not in customer-level tables.
    rules.append(
        Rule(
            "missing_customer_id",
            "Valid product line without a Customer ID (guest checkout). Kept in `lines_all` "
            "for sales analysis; excluded from `transactions` (recommendations, RFM, CLV).",
            _count(con, "SELECT count(*) FROM lines_all WHERE customer_id IS NULL"),
        )
    )
    con.execute(
        "CREATE TABLE transactions AS SELECT * FROM lines_all WHERE customer_id IS NOT NULL"
    )

    con.execute(
        """
        CREATE TABLE products AS
        SELECT stock_code, any_value(description) AS description,
               count(*) AS n_lines, count(DISTINCT customer_id) AS n_customers,
               sum(quantity) AS units, round(median(price), 2) AS median_price,
               min(invoice_date) AS first_sold, max(invoice_date) AS last_sold
        FROM lines_all GROUP BY stock_code ORDER BY stock_code
        """
    )
    con.execute(
        """
        CREATE TABLE customers AS
        SELECT customer_id, mode(country) AS country,
               count(DISTINCT invoice) AS n_orders, round(sum(revenue), 2) AS revenue,
               min(invoice_date) AS first_purchase, max(invoice_date) AS last_purchase
        FROM transactions GROUP BY customer_id ORDER BY customer_id
        """
    )
    rules.append(
        Rule(
            "final_transactions",
            "Rows in the clean `transactions` table.",
            _count(con, "SELECT count(*) FROM transactions"),
        )
    )

    for t in TABLES:
        con.execute(f"COPY {t} TO '{out_dir / (t + '.parquet')}' (FORMAT parquet)")
    if db_path is not None:
        build_duckdb(out_dir, db_path)
    con.close()
    return rules


def build_duckdb(parquet_dir: Path, db_path: Path) -> None:
    """Materialise the Parquet tables into one DuckDB file (used by the API and SQL tool)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    with duckdb.connect(str(db_path)) as con:
        for t in TABLES:
            con.execute(
                f"CREATE TABLE {t} AS SELECT * FROM read_parquet('{parquet_dir / t}.parquet')"
            )


def rules_markdown(rules: list[Rule]) -> str:
    lines = ["| Rule | What it does | Rows affected |", "|---|---|---:|"]
    lines += [f"| `{r.name}` | {r.description} | {r.rows:,} |" for r in rules]
    return "\n".join(lines)


def write_decisions(rules: list[Rule], path: Path = DOCS_DIR / "DECISIONS.md") -> None:
    start, end = "<!-- CLEANING-RULES:START -->", "<!-- CLEANING-RULES:END -->"
    block = f"{start}\n{rules_markdown(rules)}\n{end}"
    text = path.read_text() if path.exists() else "# Decisions\n"
    if start in text:
        text = re.sub(f"{start}.*?{end}", lambda _: block, text, flags=re.S)
    else:
        text += f"\n## Data cleaning rules (applied in this order)\n\n{block}\n"
    path.write_text(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = clean()
    print(rules_markdown(result))
    write_decisions(result)
