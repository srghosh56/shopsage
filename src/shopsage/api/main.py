"""FastAPI app: customer profiles, recommendations with evidence, LLM explanations, assistant."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from shopsage.llm.explain import Explainer
from shopsage.service import NotFound, ShopSageService, get_service

Mode = Literal["discover", "buy_again"]

app = FastAPI(
    title="ShopSage API",
    version="0.1.0",
    description="Recommendations and customer analytics on UCI Online Retail II.",
)


@lru_cache
def get_explainer() -> Explainer:
    return Explainer(get_service().products.description.tolist())


def svc() -> ShopSageService:
    return get_service()


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotFound as e:
        raise HTTPException(status_code=404, detail=str(e.args[0])) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/customers")
def customers(n: int = Query(20, ge=1, le=200), s: ShopSageService = Depends(svc)) -> list[int]:
    """Example customer ids (highest revenue first)."""
    return s.sample_customers(n)


@app.get("/customers/{customer_id}")
def profile(customer_id: int, s: ShopSageService = Depends(svc)) -> dict[str, Any]:
    """RFM, segment, 12-month CLV and top items."""
    return _guard(s.customer_profile, customer_id)


@app.get("/customers/{customer_id}/recommendations")
def recommendations(
    customer_id: int,
    k: int = Query(10, ge=1, le=50),
    mode: Mode = "discover",
    s: ShopSageService = Depends(svc),
) -> dict[str, Any]:
    """Ranked items with evidence: similar past purchases, co-purchase counts, popularity."""
    return _guard(s.recommend, customer_id, k, mode)


@app.get("/customers/{customer_id}/explanations")
def explanations(
    customer_id: int,
    k: int = Query(5, ge=1, le=20),
    mode: Mode = "discover",
    s: ShopSageService = Depends(svc),
) -> dict[str, Any]:
    """Recommendations plus a 1-2 sentence explanation each, written by the LLM from the
    evidence only. Explanations naming products or numbers absent from the evidence are
    replaced by a template (`source = template`)."""
    recs = _guard(s.recommend, customer_id, k, mode)
    ex = get_explainer()
    for item in recs["items"]:
        item["explanation"] = ex.explain(item)
    return recs


@app.get("/items/{stock_code}/similar")
def similar(
    stock_code: str, k: int = Query(10, ge=1, le=50), s: ShopSageService = Depends(svc)
) -> dict[str, Any]:
    return _guard(s.similar_items, stock_code, k)


@app.get("/segments")
def segments(s: ShopSageService = Depends(svc)) -> dict[str, Any]:
    return s.segment_summary()


@app.get("/models/comparison")
def model_comparison(s: ShopSageService = Depends(svc)) -> dict[str, Any]:
    """Test-window metrics for every recommender (from `make eval`)."""
    return s.model_comparison


@app.get("/clv/segments")
def clv_segments(s: ShopSageService = Depends(svc)) -> dict[str, Any]:
    """12-month CLV and P(alive) aggregated by segment."""
    return s.clv_by_segment()


@app.get("/clv/validation")
def clv_validation(s: ShopSageService = Depends(svc)) -> dict[str, Any]:
    return s.clv_validation


class SearchHit(BaseModel):
    stock_code: str
    description: str
    score: float


@app.get("/products/search", response_model=list[SearchHit])
def product_search(q: str = Query(..., min_length=2), k: int = Query(10, ge=1, le=50)):
    from shopsage.search.products import get_search

    return get_search().search(q, k)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list)


@app.post("/assistant/chat")
def assistant_chat(req: ChatRequest) -> dict[str, Any]:
    """Tool-using assistant (LangGraph). Returns the answer and the tool calls it made."""
    from shopsage.agent.graph import run_agent

    return run_agent(req.message, req.history)
