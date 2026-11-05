"""Customer lifetime value: BG/NBD (purchases) + Gamma-Gamma (spend) with pymc-marketing.

1. Holdout validation: fit on orders up to 2011-06-09, predict each customer's purchases and
   spend over the next 183 days (to 2011-12-09), compare with what actually happened and with a
   naive baseline (the customer's past rate x past average order value).
2. Serving: refit on all data and predict 12-month CLV, P(alive) and expected orders per
   customer. Written to artifacts/clv.parquet and the DuckDB table `customer_clv`.

Fitted by MAP (maximum a posteriori) for speed and determinism; time unit = days.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pymc_marketing.clv import BetaGeoModel, GammaGammaModel  # noqa: E402
from pymc_marketing.clv.utils import rfm_summary, rfm_train_test_split  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from shopsage.config import ARTIFACTS_DIR, DOCS_DIR, DUCKDB_PATH, PROCESSED_DIR, SEED  # noqa: E402
from shopsage.reporting import md_table, write_section  # noqa: E402

log = logging.getLogger(__name__)
CALIBRATION_END = "2011-06-09"
HOLDOUT_END = "2011-12-09"
HORIZON_MONTHS = 12


def orders_table(tx: pd.DataFrame) -> pd.DataFrame:
    """One row per order: customer, timestamp, order value."""
    return (
        tx.groupby(["customer_id", "invoice"])
        .agg(date=("invoice_date", "min"), value=("revenue", "sum"))
        .reset_index()
    )


def fit_models(summary: pd.DataFrame) -> tuple[BetaGeoModel, GammaGammaModel]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bg = BetaGeoModel()
        bg.fit(
            summary[["customer_id", "frequency", "recency", "T"]],
            method="map",
            progressbar=False,
            random_seed=SEED,
        )
        repeat = summary[(summary.frequency > 0) & (summary.monetary_value > 0)]
        gg = GammaGammaModel()
        gg.fit(
            repeat[["customer_id", "frequency", "monetary_value"]],
            method="map",
            progressbar=False,
            random_seed=SEED,
        )
    return bg, gg


def _flat(x) -> np.ndarray:
    """Posterior mean over chain/draw as a 1-D array (MAP has one chain and one draw)."""
    return np.asarray(x.mean(("chain", "draw"))).reshape(-1)


def predict(
    bg: BetaGeoModel, gg: GammaGammaModel, summary: pd.DataFrame, days: int
) -> pd.DataFrame:
    """Expected orders, expected spend per order and expected revenue over `days`.

    Gamma-Gamma's population mean is v*p/(q-1), which is undefined when the fit gives q <= 1
    (can happen on small samples). Then, and for any non-positive estimate, the expected order
    value falls back to the empirical mean of repeat customers' average order value.
    """
    data = summary[["customer_id", "frequency", "recency", "T", "monetary_value"]].copy()
    data["monetary_value"] = data.monetary_value.clip(lower=0.01)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        purchases = _flat(bg.expected_purchases(data=data, future_t=days))
        p_alive = _flat(bg.expected_probability_alive(data=data))
        spend = _flat(gg.expected_customer_spend(data=data))
    fallback = float(summary.loc[summary.frequency > 0, "monetary_value"].mean())
    spend = np.where(np.isfinite(spend) & (spend > 0), spend, fallback)
    return pd.DataFrame(
        {
            "customer_id": data.customer_id.to_numpy(),
            "expected_orders": purchases,
            "p_alive": p_alive,
            "expected_order_value": spend,
            "expected_revenue": purchases * spend,
        }
    )


def error_metrics(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(np.mean(np.abs(pred - actual))),
        "RMSE": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "total_error_pct": float((pred.sum() - actual.sum()) / actual.sum() * 100),
        "spearman": float(spearmanr(pred, actual).statistic),
    }


def top_share(pred: np.ndarray, actual: np.ndarray, q: float = 0.2) -> float:
    """Share of actual holdout revenue earned by the customers ranked top-q by prediction."""
    n = int(len(pred) * q)
    idx = np.argsort(-pred)[:n]
    return float(actual[idx].sum() / actual.sum())


def holdout_validation(orders: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    split = rfm_train_test_split(
        orders,
        "customer_id",
        "date",
        train_period_end=CALIBRATION_END,
        test_period_end=HOLDOUT_END,
        monetary_value_col="value",
    )
    days = int(split.test_T.iloc[0])
    bg, gg = fit_models(split)
    pred = predict(bg, gg, split, days)
    hold = orders[
        (orders.date > pd.Timestamp(CALIBRATION_END) + pd.Timedelta(days=1))
        & (orders.date <= pd.Timestamp(HOLDOUT_END) + pd.Timedelta(days=1))
    ]
    actual_rev = hold.groupby("customer_id").value.sum()
    df = pred.merge(split[["customer_id", "frequency", "T", "test_frequency"]], on="customer_id")
    df["actual_orders"] = df.test_frequency
    df["actual_revenue"] = df.customer_id.map(actual_rev).fillna(0.0)

    # naive baseline: repeat-purchase rate so far x past average order value
    calib = orders[orders.date < pd.Timestamp(CALIBRATION_END) + pd.Timedelta(days=1)]
    aov = calib.groupby("customer_id").value.mean()
    df["naive_orders"] = df.frequency / df["T"].clip(lower=1) * days
    df["naive_revenue"] = df.naive_orders * df.customer_id.map(aov).fillna(0.0)

    res = {
        "calibration_end": CALIBRATION_END,
        "holdout_end": HOLDOUT_END,
        "holdout_days": days,
        "customers": len(df),
        "bgnbd_params": {k: float(v) for k, v in bg.fit_summary().items()},
        "gg_params": {k: float(v) for k, v in gg.fit_summary().items()},
        "orders": {
            "bgnbd": error_metrics(df.expected_orders.to_numpy(), df.actual_orders.to_numpy()),
            "naive": error_metrics(df.naive_orders.to_numpy(), df.actual_orders.to_numpy()),
            "actual_total": float(df.actual_orders.sum()),
        },
        "revenue": {
            "bgnbd_gg": error_metrics(df.expected_revenue.to_numpy(), df.actual_revenue.to_numpy()),
            "naive": error_metrics(df.naive_revenue.to_numpy(), df.actual_revenue.to_numpy()),
            "actual_total": float(df.actual_revenue.sum()),
            "top20_share_bgnbd_gg": top_share(
                df.expected_revenue.to_numpy(), df.actual_revenue.to_numpy()
            ),
            "top20_share_naive": top_share(
                df.naive_revenue.to_numpy(), df.actual_revenue.to_numpy()
            ),
        },
    }
    return res, df


def plot_holdout(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    b = df.assign(bucket=df.frequency.clip(upper=10).astype(int))
    g = b.groupby("bucket")[["actual_orders", "expected_orders", "naive_orders"]].mean()
    axes[0].plot(g.index, g.actual_orders, "o-", label="actual")
    axes[0].plot(g.index, g.expected_orders, "s--", label="BG/NBD")
    axes[0].plot(g.index, g.naive_orders, "^:", label="naive rate")
    axes[0].set_xlabel("repeat purchases in calibration (10 = 10+)")
    axes[0].set_ylabel("mean purchases in holdout")
    axes[0].set_title("Holdout purchases by calibration frequency")
    axes[0].legend()
    d = df.assign(decile=pd.qcut(df.expected_revenue.rank(method="first"), 10, labels=False) + 1)
    g2 = d.groupby("decile")[["actual_revenue", "expected_revenue"]].mean()
    axes[1].bar(g2.index - 0.2, g2.actual_revenue, width=0.4, label="actual")
    axes[1].bar(g2.index + 0.2, g2.expected_revenue, width=0.4, label="BG/NBD + Gamma-Gamma")
    axes[1].set_xlabel("predicted-revenue decile (10 = highest)")
    axes[1].set_ylabel("mean holdout revenue (£)")
    axes[1].set_title("Holdout revenue by predicted decile")
    axes[1].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def serve_clv(orders: pd.DataFrame, db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    """Fit on all data and predict the next 12 months for every customer."""
    summary = rfm_summary(
        orders,
        "customer_id",
        "date",
        monetary_value_col="value",
        observation_period_end=HOLDOUT_END,
    )
    bg, gg = fit_models(summary)
    pred = predict(bg, gg, summary, days=365)
    pred = pred.rename(
        columns={"expected_orders": "expected_orders_12m", "expected_revenue": "clv_12m"}
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(ARTIFACTS_DIR / "clv.parquet", index=False)
    with duckdb.connect(str(db_path)) as con:
        con.register("p", pred)
        con.execute("CREATE OR REPLACE TABLE customer_clv AS SELECT * FROM p")
    return pred


def _fmt(m: dict[str, float]) -> dict[str, str]:
    return {
        "MAE": f"{m['MAE']:.2f}",
        "RMSE": f"{m['RMSE']:.2f}",
        "total error": f"{m['total_error_pct']:+.1f}%",
        "Spearman": f"{m['spearman']:.3f}",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tx = pd.read_parquet(PROCESSED_DIR / "transactions.parquet")
    orders = orders_table(tx)
    res, df = holdout_validation(orders)
    plot_holdout(df, DOCS_DIR / "figures" / "clv_holdout.png")
    pred = serve_clv(orders)
    res["serving"] = {
        "customers": len(pred),
        "median_clv_12m": float(pred.clv_12m.median()),
        "total_clv_12m": float(pred.clv_12m.sum()),
    }
    (ARTIFACTS_DIR / "clv_validation.json").write_text(json.dumps(res, indent=2))
    o, r = res["orders"], res["revenue"]
    rows = [
        {"target": "orders per customer", "model": "BG/NBD", **_fmt(o["bgnbd"])},
        {"target": "orders per customer", "model": "naive rate", **_fmt(o["naive"])},
        {
            "target": "revenue per customer (£)",
            "model": "BG/NBD + Gamma-Gamma",
            **_fmt(r["bgnbd_gg"]),
        },
        {"target": "revenue per customer (£)", "model": "naive rate × AOV", **_fmt(r["naive"])},
    ]
    md = f"""
Calibration: orders up to {CALIBRATION_END}; holdout: the next {res["holdout_days"]:.0f} days to
{HOLDOUT_END}; {res["customers"]:,} customers who bought in calibration. Same-day orders count as
one purchase (time unit = day). Fitted by MAP with `pymc-marketing`. Naive baseline = the
customer's calibration repeat rate × holdout length (× their average order value for revenue).
Generated by `python -m shopsage.analytics.clv`.

{md_table(rows, ["target", "model", "MAE", "RMSE", "total error", "Spearman"])}

- Actual holdout totals: {o["actual_total"]:,.0f} purchases, £{r["actual_total"]:,.0f} revenue.
- Customers ranked in the model's top 20% earned **{r["top20_share_bgnbd_gg"]:.1%}** of actual
  holdout revenue (naive ranking: {r["top20_share_naive"]:.1%}).
- Serving model (refit on all data): median 12-month CLV £{res["serving"]["median_clv_12m"]:,.0f},
  total £{res["serving"]["total_clv_12m"]:,.0f} across {res["serving"]["customers"]:,} customers.

![CLV holdout](figures/clv_holdout.png)
"""
    write_section("CLV", md)
    print(json.dumps({k: res[k] for k in ["orders", "revenue"]}, indent=2))


if __name__ == "__main__":
    main()
