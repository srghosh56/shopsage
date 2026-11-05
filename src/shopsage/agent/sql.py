"""Read-only SQL over the DuckDB tables for the assistant.

Defence in depth: (1) sqlglot parses the query: exactly one statement, a SELECT (or set
operation of SELECTs), no write/DDL/PRAGMA/file-reading nodes, only whitelisted tables; (2) a
LIMIT is added or capped; (3) the DuckDB connection is opened read-only; (4) a timer interrupts
queries that run too long.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

from shopsage.config import DUCKDB_PATH

MAX_ROWS = 100
TIMEOUT_S = 5.0

TABLES: dict[str, str] = {
    "transactions": "invoice, stock_code, description, quantity, invoice_date, price, revenue, "
    "customer_id, country — one row per purchased product line (customers with an ID only)",
    "lines_all": "same columns as transactions, also guest checkouts (customer_id NULL)",
    "returns": "invoice, stock_code, quantity, invoice_date, price, customer_id, country — "
    "cancelled lines",
    "products": "stock_code, description, n_lines, n_customers, units, median_price, "
    "first_sold, last_sold",
    "customers": "customer_id, country, n_orders, revenue, first_purchase, last_purchase",
    "customer_segments": "customer_id, recency, frequency, monetary, cluster, segment",
    "segment_summary": "segment, customers, median_recency_days, median_orders, median_revenue, "
    "total_revenue, share_customers, share_revenue, action",
    "customer_clv": "customer_id, expected_orders_12m, p_alive, expected_order_value, clv_12m",
}

_BLOCKED_NAMES = [
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "Command",
    "Copy",
    "Pragma",
    "Set",
    "Use",
    "Merge",
    "TruncateTable",
    "Attach",
    "Detach",
    "Transaction",
    "Commit",
    "Rollback",
    "Grant",
    "Into",
    "LoadData",
    "Export",
    "Install",
]
BLOCKED = tuple(getattr(exp, n) for n in _BLOCKED_NAMES if hasattr(exp, n))
BLOCKED_FUNCS = {
    "read_csv",
    "read_csv_auto",
    "read_parquet",
    "read_json",
    "read_json_auto",
    "read_text",
    "read_blob",
    "glob",
    "parquet_scan",
    "csv_scan",
    "sniff_csv",
    "query",
    "query_table",
    "getenv",
    "read_ndjson",
}


class UnsafeQuery(ValueError):
    pass


def validate(query: str, max_rows: int = MAX_ROWS) -> str:
    """Return a safe, LIMIT-ed version of `query` or raise UnsafeQuery."""
    try:
        statements = [s for s in sqlglot.parse(query, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeQuery(f"could not parse SQL: {str(e)[:200]}") from e
    if len(statements) != 1:
        raise UnsafeQuery("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, exp.Select | exp.SetOperation):
        raise UnsafeQuery(f"only SELECT queries are allowed, got {type(tree).__name__}")
    for node in tree.walk():
        if isinstance(node, BLOCKED):
            raise UnsafeQuery(f"{type(node).__name__} is not allowed")
        if isinstance(node, exp.Func):
            name = (node.sql_name() if not isinstance(node, exp.Anonymous) else node.name).lower()
            if name in BLOCKED_FUNCS:
                raise UnsafeQuery(f"function {name} is not allowed")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        name = t.name.lower()
        if not name and t.this is not None and not isinstance(t.this, exp.Identifier):
            raise UnsafeQuery("table functions are not allowed")
        if name not in TABLES and name not in ctes:
            raise UnsafeQuery(f"unknown table {t.name!r}; allowed: {sorted(TABLES)}")
        if t.args.get("db") or t.args.get("catalog"):
            raise UnsafeQuery("schema-qualified tables are not allowed")
    limit = tree.args.get("limit")
    current = None
    if isinstance(limit, exp.Limit):
        try:
            current = int(limit.expression.name)
        except (AttributeError, ValueError):
            current = None
    if current is None or current > max_rows:
        tree = tree.limit(max_rows)  # type: ignore[attr-defined]
    return tree.sql(dialect="duckdb")


def run_readonly(
    query: str, db_path: Path = DUCKDB_PATH, timeout_s: float = TIMEOUT_S
) -> dict[str, Any]:
    safe = validate(query)
    con = duckdb.connect(str(db_path), read_only=True)
    timer = threading.Timer(timeout_s, con.interrupt)
    timer.start()
    try:
        cur = con.execute(safe)
        cols = [d[0] for d in cur.description or []]
        rows = cur.fetchall()
    except duckdb.InterruptException as e:
        raise UnsafeQuery(f"query timed out after {timeout_s:.0f}s") from e
    finally:
        timer.cancel()
        con.close()
    return {
        "sql_executed": safe,
        "columns": cols,
        "rows": [[_jsonable(v) for v in r] for r in rows],
        "row_count": len(rows),
    }


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, int | float | str | bool):
        return round(v, 4) if isinstance(v, float) else v
    return str(v)


def schema_text() -> str:
    return "\n".join(f"- {t}: {cols}" for t, cols in TABLES.items())
