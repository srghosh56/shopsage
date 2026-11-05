"""The assistant's tools. Plain functions over the service layer, shared by LangGraph and MCP.

Each returns a JSON string; errors come back as {"error": ...} so the model can recover.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import Any

from shopsage.agent.sql import UnsafeQuery, run_readonly, schema_text
from shopsage.llm.explain import Explainer
from shopsage.service import NotFound, get_service

MAX_CHARS = 6000


def _json(fn: Callable[..., Any]) -> Callable[..., str]:
    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            out = fn(*args, **kwargs)
        except (NotFound, UnsafeQuery, ValueError) as e:
            out = {"error": str(e.args[0]) if e.args else type(e).__name__}
        text = json.dumps(out, default=str)
        return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + '..."(truncated)"'

    # the wrapper returns JSON text, whatever the wrapped function returns
    wrapper.__annotations__ = {**fn.__annotations__, "return": "str"}
    wrapper.__signature__ = inspect.signature(fn).replace(return_annotation=str)  # type: ignore[attr-defined]
    return wrapper


@lru_cache
def _explainer() -> Explainer:
    return Explainer(get_service().products.description.tolist())


def _slim(item: dict[str, Any]) -> dict[str, Any]:
    ev = item["evidence"]
    return {
        "rank": item["rank"],
        "stock_code": item["stock_code"],
        "description": item["description"],
        "median_price_gbp": item["median_price"],
        "buyers": ev["popularity"]["buyers"],
        "similar_past_purchases": [
            {"description": s["description"], "co_purchase_customers": s["co_purchase_customers"]}
            for s in ev.get("similar_past_purchases", [])
        ],
        **({"your_history": ev["your_history"]} if "your_history" in ev else {}),
    }


@_json
def get_customer_profile(customer_id: int) -> dict[str, Any]:
    """Profile of one customer: country, first/last purchase, RFM (recency days, number of
    orders, revenue in GBP), segment, 12-month CLV, probability still active, top items."""
    return get_service().customer_profile(customer_id)


@_json
def get_recommendations(customer_id: int, k: int = 5, mode: str = "discover") -> dict[str, Any]:
    """Recommended products for a customer with evidence. mode='discover' = new items they
    haven't bought (ALS model); mode='buy_again' = items they are likely to re-order."""
    recs = get_service().recommend(customer_id, min(int(k), 20), mode)
    return {
        **{k_: v for k_, v in recs.items() if k_ != "items"},
        "items": [_slim(i) for i in recs["items"]],
    }


@_json
def explain_recommendations(customer_id: int, k: int = 3, mode: str = "discover") -> dict[str, Any]:
    """Recommendations for a customer, each with a short evidence-based explanation."""
    recs = get_service().recommend(customer_id, min(int(k), 10), mode)
    ex = _explainer()
    return {
        "customer_id": recs["customer_id"],
        "mode": mode,
        "items": [
            {
                "description": i["description"],
                "stock_code": i["stock_code"],
                "explanation": ex.explain(i)["text"],
            }
            for i in recs["items"]
        ],
    }


@_json
def find_similar_items(stock_code: str, k: int = 5) -> dict[str, Any]:
    """Products most often bought by the same customers as the given stock code (cosine
    similarity of buyers), with co-purchase counts."""
    return get_service().similar_items(stock_code, min(int(k), 20))


@_json
def search_products(query: str, k: int = 5) -> dict[str, Any]:
    """Find products by description (keyword BM25 + semantic embeddings, fused). Returns stock
    codes and descriptions. Use this to turn a product name into a stock code."""
    from shopsage.search.products import get_search

    return {"query": query, "results": get_search().search(query, min(int(k), 20))}


@_json
def get_segment_summary(segment: str = "") -> dict[str, Any]:
    """Customer segments (RFM + K-means): size, medians, revenue share and suggested action.
    Leave segment empty for all segments."""
    return get_service().segment_summary(segment or None)


@_json
def run_sql(query: str) -> dict[str, Any]:
    """Run ONE read-only DuckDB SELECT over the shop's tables (max 100 rows, 5 s timeout).
    Use for counts, totals, rankings and date questions. Revenue is in GBP. Data covers
    2009-12-01 to 2011-12-09."""
    return run_readonly(query)


run_sql.__doc__ = (run_sql.__doc__ or "") + "\nTables:\n" + schema_text()

TOOLS: list[Callable[..., str]] = [
    get_customer_profile,
    get_recommendations,
    explain_recommendations,
    find_similar_items,
    search_products,
    get_segment_summary,
    run_sql,
]
