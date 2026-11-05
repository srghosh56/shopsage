"""Tune every recommender on the validation window and log each config to MLflow.

Fit on data before 2011-06-01, score on 2011-06-01..2011-08-31, choose by NDCG@10 separately
for the `all` and `new` protocols. Writes artifacts/best_params.json, then fits the chosen
models on *all* data for serving (artifacts/models/*.pkl).
"""

from __future__ import annotations

import itertools
import json
import logging
import pickle
from collections.abc import Callable, Iterable
from typing import Any

import mlflow
import pandas as pd

from shopsage.config import ARTIFACTS_DIR, MLFLOW_URI, VAL_START
from shopsage.eval.runner import PROTOCOLS, evaluate_model
from shopsage.eval.split import make_validation_split
from shopsage.recsys.base import Recommender
from shopsage.recsys.baselines import BuyAgain, Popularity, RecentPopularity
from shopsage.recsys.data import Interactions, load_transactions
from shopsage.recsys.models import ALS, ItemKNN

log = logging.getLogger(__name__)
K = 10
SELECT_BY = f"ndcg@{K}"


def grid(**axes: Iterable[Any]) -> list[dict[str, Any]]:
    keys = list(axes)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*axes.values())]


SEARCH_SPACE: dict[str, tuple[Callable[..., Recommender], list[dict[str, Any]]]] = {
    "popularity": (Popularity, [{}]),
    "recent_popularity": (RecentPopularity, grid(window_days=[7, 14, 30, 60, 90])),
    "buy_again": (BuyAgain, grid(half_life_days=[14, 30, 90, 180, 365])),
    "item_knn": (
        ItemKNN,
        grid(
            n_neighbors=[5, 10, 20, 50, 100, 200, 500],
            shrink=[0, 10, 50],
            weighting=["binary", "log"],
        ),
    ),
    "als": (
        ALS,
        grid(
            factors=[32, 64, 128, 256],
            regularization=[0.01, 0.1, 1.0, 5.0, 20.0],
            alpha=[0.25, 1.0, 5.0, 20.0],
            iterations=[20],
            weighting=["log"],
        ),
    ),
}


def mlflow_key(metric: str) -> str:
    return metric.replace("@", "_at_")


def tune(
    tx: pd.DataFrame, models: list[str] | None = None, space: dict | None = None
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return {protocol: {model: {"params": ..., "val": metrics}}} for the best config."""
    space = space or SEARCH_SPACE
    inner, val = make_validation_split(tx)
    data = Interactions.from_lines(inner, end=pd.Timestamp(VAL_START))
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("shopsage-recsys-tuning")
    best: dict[str, dict[str, dict[str, Any]]] = {p: {} for p in PROTOCOLS}
    for name in models or list(space):
        factory, configs = space[name]
        for params in configs:
            model = factory(**params).fit(data)
            for protocol in PROTOCOLS:
                res = evaluate_model(
                    model, data, val, k=K, protocol=protocol, beyond_accuracy=False
                )
                with mlflow.start_run(run_name=f"{name}-{protocol}"):
                    mlflow.set_tags({"model": name, "protocol": protocol, "split": "validation"})
                    mlflow.log_params(params)
                    mlflow.log_metrics({mlflow_key(k): v for k, v in res.items()})
                cur = best[protocol].get(name)
                if cur is None or res[SELECT_BY] > cur["val"][SELECT_BY]:
                    best[protocol][name] = {"params": params, "val": res}
            log.info("%s %s", name, params)
    return best


def fit_for_serving(tx: pd.DataFrame, best: dict[str, dict[str, dict[str, Any]]]) -> None:
    """Fit the best `new`-protocol item-kNN and ALS and the best buy-again on all data."""
    data = Interactions.from_lines(tx)
    out = ARTIFACTS_DIR / "models"
    out.mkdir(parents=True, exist_ok=True)
    chosen = {
        "item_knn": ItemKNN(**best["new"]["item_knn"]["params"]),
        "als": ALS(**best["new"]["als"]["params"]),
        "buy_again": BuyAgain(**best["all"]["buy_again"]["params"]),
        "popularity": Popularity(),
    }
    for name, model in chosen.items():
        model.fit(data)
        with open(out / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)
    log.info("saved serving models to %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tx = load_transactions()
    best = tune(tx)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "best_params.json").write_text(json.dumps(best, indent=2, default=str))
    for protocol, models in best.items():
        for name, b in models.items():
            log.info(
                "[%s] %-18s %s=%.4f %s", protocol, name, SELECT_BY, b["val"][SELECT_BY], b["params"]
            )
    fit_for_serving(tx, best)


if __name__ == "__main__":
    main()
