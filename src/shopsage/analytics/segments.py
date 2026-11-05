"""RFM features + K-means customer segments with business names.

Recency = days since the last order (snapshot = day after the last invoice), frequency = number
of orders, monetary = gross revenue. Features are log-transformed and standardised; k is chosen
by silhouette score in [4, 6]. Each cluster is named from its centroid (see `name_segment`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from shopsage.config import ARTIFACTS_DIR, DUCKDB_PATH, PROCESSED_DIR, SEED
from shopsage.reporting import md_table, write_section

log = logging.getLogger(__name__)
FEATURES = ["recency", "frequency", "monetary"]
K_RANGE = range(4, 7)

ACTIONS = {
    "Champions": "Reward and ask for referrals; early access to new ranges.",
    "Loyal regulars": "Cross-sell complementary items; volume discounts.",
    "Promising new": "Onboarding offers to secure the second and third order.",
    "Needs attention": "Personalised reminders with buy-again lists.",
    "At risk": "Win-back campaign before they lapse: they used to buy often.",
    "Hibernating": "Low-cost reactivation only; don't over-invest.",
}


def rfm_table(tx: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    snapshot = snapshot or (tx.invoice_date.max().normalize() + pd.Timedelta(days=1))
    g = tx.groupby("customer_id")
    rfm = pd.DataFrame(
        {
            "recency": (snapshot - g.invoice_date.max()).dt.days,
            "frequency": g.invoice.nunique(),
            "monetary": g.revenue.sum().round(2),
        }
    )
    return rfm.reset_index()


def name_segment(r: float, f: float, m: float) -> str:
    """Name a cluster from its centroid z-scores (r = recency z, lower = more recent)."""
    active, frequent = r < 0, (f + m) / 2 > 0
    if active and frequent:
        return "Champions" if (f + m) / 2 > 1.0 else "Loyal regulars"
    if active:
        return "Promising new" if r < -0.5 else "Needs attention"
    if frequent:
        return "At risk"
    return "Hibernating" if r > 0.5 else "Needs attention"


def segment(rfm: pd.DataFrame, k: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Add `cluster` and `segment` columns. Returns (customers, segment summary, diagnostics)."""
    x = np.column_stack(
        [np.log1p(rfm.recency), np.log1p(rfm.frequency), np.log1p(rfm.monetary.clip(lower=0))]
    )
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    scores = {}
    for kk in [k] if k else K_RANGE:
        labels = KMeans(kk, n_init=10, random_state=SEED).fit_predict(z)
        sample = min(len(z), 5000)
        scores[kk] = float(silhouette_score(z, labels, sample_size=sample, random_state=SEED))
    best_k = max(scores, key=lambda kk: scores[kk])
    km = KMeans(best_k, n_init=10, random_state=SEED).fit(z)
    out = rfm.copy()
    out["cluster"] = km.labels_
    names = {}
    for c, (r, f, m) in enumerate(km.cluster_centers_):
        names[c] = name_segment(r, f, m)
    # if two clusters get the same name, the one with higher value keeps it, others get a suffix
    order = np.argsort(-(km.cluster_centers_[:, 1] + km.cluster_centers_[:, 2]))
    seen: dict[str, int] = {}
    for c in order:
        base = names[c]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[c] = f"{base} ({'lower value' if seen[base] == 2 else seen[base]})"
    out["segment"] = out.cluster.map(names)
    summary = (
        out.groupby("segment")
        .agg(
            customers=("customer_id", "size"),
            median_recency_days=("recency", "median"),
            median_orders=("frequency", "median"),
            median_revenue=("monetary", "median"),
            total_revenue=("monetary", "sum"),
        )
        .reset_index()
    )
    summary["share_customers"] = summary.customers / summary.customers.sum()
    summary["share_revenue"] = summary.total_revenue / summary.total_revenue.sum()
    summary["action"] = summary.segment.map(lambda s: ACTIONS[s.split(" (")[0]])
    summary = summary.sort_values("median_revenue", ascending=False).reset_index(drop=True)
    diag = {"k": best_k, "silhouette": scores}
    return out, summary, diag


def save(customers: pd.DataFrame, summary: pd.DataFrame, db_path: Path = DUCKDB_PATH) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    customers.to_parquet(ARTIFACTS_DIR / "customer_segments.parquet", index=False)
    summary.to_parquet(ARTIFACTS_DIR / "segment_summary.parquet", index=False)
    with duckdb.connect(str(db_path)) as con:
        con.register("c", customers)
        con.register("s", summary)
        con.execute("CREATE OR REPLACE TABLE customer_segments AS SELECT * FROM c")
        con.execute("CREATE OR REPLACE TABLE segment_summary AS SELECT * FROM s")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tx = pd.read_parquet(PROCESSED_DIR / "transactions.parquet")
    customers, summary, diag = segment(rfm_table(tx))
    save(customers, summary)
    (ARTIFACTS_DIR / "segments_diag.json").write_text(json.dumps(diag, indent=2))
    sil = ", ".join(f"k={k}: {v:.3f}" for k, v in diag["silhouette"].items())
    rows = summary.assign(
        median_recency_days=summary.median_recency_days.round(0).astype(int).astype(str),
        median_orders=summary.median_orders.round(1).astype(str),
        median_revenue=summary.median_revenue.map(lambda v: f"£{v:,.0f}"),
        share_customers=summary.share_customers.map(lambda v: f"{v:.1%}"),
        share_revenue=summary.share_revenue.map(lambda v: f"{v:.1%}"),
    ).to_dict("records")
    cols = [
        "segment",
        "customers",
        "median_recency_days",
        "median_orders",
        "median_revenue",
        "share_customers",
        "share_revenue",
        "action",
    ]
    md = f"""
RFM on all {len(customers):,} customers (snapshot {tx.invoice_date.max().date() + pd.Timedelta(days=1)}),
log + standardised features, K-means with k chosen by silhouette ({sil}) → **k = {diag["k"]}**.
Names come from each centroid's position (rule in `analytics/segments.py::name_segment`).
Generated by `python -m shopsage.analytics.segments`.

{md_table(rows, cols)}
"""
    write_section("SEGMENTS", md)
    print(summary.to_string())


if __name__ == "__main__":
    main()
