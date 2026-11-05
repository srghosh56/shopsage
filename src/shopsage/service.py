"""One in-process service used by the API, the assistant's tools and the MCP server.

Loads the serving artifacts written by `make train` (models fitted on all data, segments, CLV)
and returns plain dicts, so every consumer sees the same numbers.
"""

from __future__ import annotations

import json
import pickle
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from shopsage.config import ARTIFACTS_DIR, DUCKDB_PATH
from shopsage.recsys.base import Recommender
from shopsage.recsys.models import ItemKNN

MODES = ("discover", "buy_again")


class NotFound(KeyError):
    pass


class ShopSageService:
    def __init__(self, artifacts: Path = ARTIFACTS_DIR, db_path: Path = DUCKDB_PATH) -> None:
        self.artifacts = artifacts
        self.db_path = db_path
        models = artifacts / "models"
        self.models: dict[str, Recommender] = {}
        for name in ("als", "item_knn", "buy_again", "popularity"):
            with open(models / f"{name}.pkl", "rb") as f:
                self.models[name] = pickle.load(f)
        knn = self.models["item_knn"]
        assert isinstance(knn, ItemKNN)
        self.knn = knn
        self.data = knn.data
        self.segments = pd.read_parquet(artifacts / "customer_segments.parquet").set_index(
            "customer_id"
        )
        self.segment_table = pd.read_parquet(artifacts / "segment_summary.parquet")
        self.clv = pd.read_parquet(artifacts / "clv.parquet").set_index("customer_id")
        with self.connect() as con:
            self.products = con.execute("SELECT * FROM products").df().set_index("stock_code")
            self.customers = con.execute("SELECT * FROM customers").df().set_index("customer_id")
        self._pop_rank = pd.Series(self.data.item_buyers).rank(ascending=False, method="min")

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path), read_only=True)

    # ---------- products ----------
    def product(self, code: str) -> dict[str, Any]:
        code = str(code).upper()
        if code not in self.products.index:
            raise NotFound(f"unknown stock code {code}")
        p = self.products.loc[code].to_dict()
        return {
            "stock_code": code,
            "description": str(p["description"]),
            "median_price": float(p["median_price"]),
            "n_customers": int(p["n_customers"]),
        }

    def describe(self, code: str) -> str:
        return str(self.products.description.get(code, code))

    def _popularity(self, i: int) -> dict[str, Any]:
        return {
            "buyers": int(self.data.item_buyers[i]),
            "rank": int(self._pop_rank.iloc[i]),
            "catalog_size": int(self.data.n_items),
        }

    # ---------- customers ----------
    def _uidx(self, customer_id: int) -> int:
        idx = self.data.user_index.get(int(customer_id))
        if idx is None:
            raise NotFound(f"unknown customer {customer_id}")
        return idx

    def customer_profile(self, customer_id: int) -> dict[str, Any]:
        cid = int(customer_id)
        if cid not in self.customers.index:
            raise NotFound(f"unknown customer {customer_id}")
        c, s, v = (
            self.customers.loc[cid].to_dict(),
            self.segments.loc[cid].to_dict(),
            self.clv.loc[cid].to_dict(),
        )
        u = self._uidx(cid)
        row = self.data.counts[u]
        top = np.argsort(-row.data)[:5]
        return {
            "customer_id": cid,
            "country": str(c["country"]),
            "first_purchase": str(pd.Timestamp(c["first_purchase"]).date()),
            "last_purchase": str(pd.Timestamp(c["last_purchase"]).date()),
            "rfm": {
                "recency_days": int(s["recency"]),
                "orders": int(s["frequency"]),
                "revenue_gbp": round(float(s["monetary"]), 2),
            },
            "segment": str(s["segment"]),
            "clv": {
                "clv_12m_gbp": round(float(v["clv_12m"]), 2),
                "expected_orders_12m": round(float(v["expected_orders_12m"]), 2),
                "p_alive": round(float(v["p_alive"]), 3),
                "expected_order_value_gbp": round(float(v["expected_order_value"]), 2),
            },
            "distinct_items_bought": int(row.nnz),
            "top_items": [
                {
                    "stock_code": str(self.data.item_ids[row.indices[j]]),
                    "description": self.describe(str(self.data.item_ids[row.indices[j]])),
                    "orders": int(row.data[j]),
                }
                for j in top
            ],
        }

    # ---------- recommendations ----------
    def _similar_purchases(self, u: int, i: int, n: int = 3) -> list[dict[str, Any]]:
        """The user's past items most similar to item i (item-kNN), with co-purchase counts."""
        bought = self.data.binary[u].indices
        sims = self.knn.similarity[:, [i]].toarray().ravel()[bought]
        # fall back to raw co-purchase counts if no bought item is among i's neighbours
        co = self.knn.co_counts[bought, i]
        order = np.lexsort((-co, -sims))[:n]
        out = []
        for j in order:
            if co[j] <= 0:
                continue
            code = str(self.data.item_ids[bought[j]])
            out.append(
                {
                    "stock_code": code,
                    "description": self.describe(code),
                    "co_purchase_customers": int(co[j]),
                    "similarity": round(float(sims[j]), 3),
                }
            )
        return out

    def recommend(self, customer_id: int, k: int = 10, mode: str = "discover") -> dict[str, Any]:
        """`discover`: ALS, already-bought items excluded. `buy_again`: re-order list."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        cid = int(customer_id)
        cold = cid not in self.data.user_index
        if cold:
            model_name = "popularity"
            items = self.models["popularity"].recommend(np.array([0]), k)[0]
        else:
            u = self._uidx(cid)
            model_name = "als" if mode == "discover" else "buy_again"
            items = self.models[model_name].recommend(
                np.array([u]), k, exclude_seen=mode == "discover"
            )[0]
        recs = []
        for rank, i in enumerate(items, 1):
            code = str(self.data.item_ids[i])
            ev: dict[str, Any] = {"popularity": self._popularity(int(i))}
            if not cold:
                u = self._uidx(cid)
                ev["similar_past_purchases"] = self._similar_purchases(u, int(i))
                times = int(self.data.counts[u, i])
                if times:
                    last = self.data.lines[
                        (self.data.lines.customer_id == cid) & (self.data.lines.stock_code == code)
                    ].invoice_date.max()
                    ev["your_history"] = {"times_ordered": times, "last_ordered": str(last.date())}
            recs.append({"rank": rank, **self.product(code), "evidence": ev})
        return {
            "customer_id": cid,
            "mode": mode,
            "model": model_name,
            "cold_start": cold,
            "items": recs,
        }

    def similar_items(self, stock_code: str, k: int = 10) -> dict[str, Any]:
        code = str(stock_code).upper()
        i = self.data.item_index.get(code)
        if i is None:
            raise NotFound(f"unknown stock code {code}")
        b = self.data.binary
        # full cosine (not the pruned kNN graph) so every item has k neighbours
        col = b[:, i]
        co = np.asarray((b.T @ col).toarray()).ravel()
        n = self.data.item_buyers
        sim = co / (np.sqrt(n * n[i]) + 1e-9)
        sim[i] = -1
        top = np.argsort(-sim)[:k]
        return {
            "item": self.product(code),
            "similar": [
                {
                    **self.product(str(self.data.item_ids[j])),
                    "cosine": round(float(sim[j]), 3),
                    "co_purchase_customers": int(co[j]),
                }
                for j in top
            ],
        }

    # ---------- segments / results ----------
    def segment_summary(self, segment: str | None = None) -> dict[str, Any]:
        t = self.segment_table
        if segment:
            t = t[t.segment.str.lower() == segment.lower()]
            if t.empty:
                raise NotFound(
                    f"unknown segment {segment}; known: {list(self.segment_table.segment)}"
                )
        rows = json.loads(t.to_json(orient="records"))
        return {"segments": rows}

    def clv_by_segment(self) -> dict[str, Any]:
        j = self.clv.join(self.segments[["segment"]])
        g = j.groupby("segment").agg(
            customers=("clv_12m", "size"),
            mean_clv_12m=("clv_12m", "mean"),
            median_clv_12m=("clv_12m", "median"),
            total_clv_12m=("clv_12m", "sum"),
            mean_p_alive=("p_alive", "mean"),
        )
        return {"segments": json.loads(g.round(2).reset_index().to_json(orient="records"))}

    @cached_property
    def model_comparison(self) -> dict[str, Any]:
        return json.loads((self.artifacts / "test_results.json").read_text())

    @cached_property
    def clv_validation(self) -> dict[str, Any]:
        return json.loads((self.artifacts / "clv_validation.json").read_text())

    def sample_customers(self, n: int = 20) -> list[int]:
        """A few known customers with a recommendation history, for demos."""
        top = self.segments.sort_values("monetary", ascending=False)
        return [int(c) for c in top.index[:n] if int(c) in self.data.user_index]


@lru_cache
def get_service() -> ShopSageService:
    return ShopSageService()
