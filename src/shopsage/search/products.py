"""Hybrid product search: BM25 (keywords) + sentence embeddings (meaning), merged by RRF.

Reciprocal rank fusion: score(d) = sum over rankers of 1 / (60 + rank_r(d)). It needs no score
calibration between the two rankers, which have very different scales.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from shopsage.config import ARTIFACTS_DIR, EMBEDDING_MODEL, PROCESSED_DIR

RRF_K = 60


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return scores


class ProductSearch:
    def __init__(
        self,
        products: pd.DataFrame,
        use_embeddings: bool = True,
        cache: Path | None = ARTIFACTS_DIR / "product_embeddings.npy",
    ) -> None:
        self.products = products.reset_index(drop=True)
        self.texts = self.products.description.fillna("").tolist()
        self.bm25 = BM25Okapi([tokenize(t) or ["_"] for t in self.texts])
        self.encoder = None
        self.emb: np.ndarray | None = None
        if use_embeddings:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
            if cache is not None and cache.exists():
                emb = np.load(cache)
                if emb.shape[0] == len(self.texts):
                    self.emb = emb
            if self.emb is None:
                self.emb = self.encoder.encode(
                    [t.lower() for t in self.texts], batch_size=256, normalize_embeddings=True
                )
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, self.emb)

    def bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:n]
        return [int(i) for i in top if scores[i] > 0]

    def embedding_ranking(self, query: str, n: int) -> list[int]:
        if self.encoder is None or self.emb is None:
            return []
        q = self.encoder.encode([query.lower()], normalize_embeddings=True)[0]
        return [int(i) for i in np.argsort(-(self.emb @ q))[:n]]

    def search(self, query: str, k: int = 10, mode: str = "hybrid") -> list[dict[str, Any]]:
        n = max(50, k * 5)
        rankings = []
        if mode in ("hybrid", "bm25"):
            rankings.append(self.bm25_ranking(query, n))
        if mode in ("hybrid", "embedding"):
            rankings.append(self.embedding_ranking(query, n))
        fused = sorted(rrf(rankings).items(), key=lambda kv: -kv[1])[:k]
        return [
            {
                "stock_code": str(self.products.stock_code[i]),
                "description": self.texts[i],
                "score": round(s, 5),
            }
            for i, s in fused
        ]


@lru_cache
def get_search() -> ProductSearch:
    products = pd.read_parquet(
        PROCESSED_DIR / "products.parquet", columns=["stock_code", "description"]
    )
    return ProductSearch(products)
