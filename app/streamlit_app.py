"""ShopSage Streamlit UI. Every number comes from the FastAPI service (API_URL)."""

from __future__ import annotations

import os
from typing import Any

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
BLUE = "#2a78d6"  # categorical slot 1 (single-series charts)

st.set_page_config(page_title="ShopSage", page_icon="🛍️", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def get(path: str, **params: Any) -> Any:
    r = httpx.get(f"{API_URL}{path}", params=params, timeout=180)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def post(path: str, payload: dict) -> Any:
    r = httpx.post(f"{API_URL}{path}", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def bar(df: pd.DataFrame, x: str, y: str, title: str, fmt: str = ",.0f") -> alt.Chart:
    return (
        alt.Chart(df, title=title)
        .mark_bar(color=BLUE, cornerRadiusTopRight=4, cornerRadiusBottomRight=4, size=18)
        .encode(
            x=alt.X(f"{x}:Q", title=None, axis=alt.Axis(format=fmt, grid=True, gridOpacity=0.3)),
            y=alt.Y(f"{y}:N", sort="-x", title=None),
            tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)],
        )
        .properties(height=alt.Step(30))
    )


def page_customer() -> None:
    st.header("Customer lookup")
    try:
        examples = get("/customers", n=20)
    except httpx.HTTPError as e:
        st.error(f"API not reachable at {API_URL}: {e}")
        return
    c1, c2, c3 = st.columns([2, 1, 1])
    cid = c1.selectbox(
        "Customer ID (top customers listed; type any ID)", examples, accept_new_options=True
    )
    k = c2.slider("Items", 3, 10, 5)
    explain = c3.toggle(
        "LLM explanations",
        value=True,
        help="Written by the LLM from the evidence only; checked, else a template.",
    )
    try:
        cid = int(cid)
    except (TypeError, ValueError):
        st.warning("Enter a numeric customer ID.")
        return
    prof = get(f"/customers/{cid}")
    if prof is None:
        st.warning(f"Customer {cid} not found.")
        return
    m = st.columns(5)
    m[0].metric("Segment", prof["segment"])
    m[1].metric("Orders", f"{prof['rfm']['orders']:,}")
    m[2].metric("Revenue", f"£{prof['rfm']['revenue_gbp']:,.0f}")
    m[3].metric("12-month CLV", f"£{prof['clv']['clv_12m_gbp']:,.0f}")
    m[4].metric("P(still active)", f"{prof['clv']['p_alive']:.0%}")
    st.caption(
        f"{prof['country']} · first order {prof['first_purchase']} · last order "
        f"{prof['last_purchase']} ({prof['rfm']['recency_days']} days before data end) · "
        f"{prof['distinct_items_bought']} distinct items bought"
    )

    for mode, title in [
        ("discover", "You may also like (ALS, new items)"),
        ("buy_again", "Buy again"),
    ]:
        st.subheader(title)
        path = f"/customers/{cid}/{'explanations' if explain else 'recommendations'}"
        with st.spinner("Asking the model…"):
            recs = get(path, k=k, mode=mode)
        for item in recs["items"]:
            ev = item["evidence"]
            with st.container(border=True):
                left, right = st.columns([3, 2])
                left.markdown(
                    f"**{item['rank']}. {item['description'].title()}** "
                    f"`{item['stock_code']}` · £{item['median_price']:.2f}"
                )
                if explain:
                    ex = item["explanation"]
                    left.write(ex["text"])
                    left.caption(f"source: {ex['source']} · check: {ex['check']}")
                sims = ev.get("similar_past_purchases", [])
                lines = [
                    f"Bought by {ev['popularity']['buyers']:,} customers "
                    f"(#{ev['popularity']['rank']:,} of {ev['popularity']['catalog_size']:,})"
                ]
                lines += [
                    f"You bought *{s['description'].title()}*: {s['co_purchase_customers']} "
                    f"customers bought both"
                    for s in sims
                ]
                if "your_history" in ev:
                    h = ev["your_history"]
                    lines.append(f"You ordered it {h['times_ordered']}× (last {h['last_ordered']})")
                right.markdown("\n".join(f"- {line}" for line in lines))


def page_segments() -> None:
    st.header("Customer segments (RFM + K-means)")
    seg = pd.DataFrame(get("/segments")["segments"])
    clv = pd.DataFrame(get("/clv/segments")["segments"])
    a, b = st.columns(2)
    a.altair_chart(bar(seg, "customers", "segment", "Customers per segment"), width="stretch")
    b.altair_chart(bar(seg, "share_revenue", "segment", "Share of revenue", ".0%"), width="stretch")
    show = seg[
        [
            "segment",
            "customers",
            "median_recency_days",
            "median_orders",
            "median_revenue",
            "share_revenue",
            "action",
        ]
    ].copy()
    show["share_revenue"] = (show.share_revenue * 100).round(1).astype(str) + "%"
    st.dataframe(show, hide_index=True, width="stretch")
    st.subheader("12-month CLV by segment")
    st.altair_chart(
        bar(clv, "mean_clv_12m", "segment", "Mean predicted 12-month CLV (£)"),
        width="stretch",
    )
    st.dataframe(clv, hide_index=True, width="stretch")


def page_clv() -> None:
    st.header("Customer lifetime value: holdout validation")
    v = get("/clv/validation")
    st.write(
        f"Calibration up to **{v['calibration_end']}**, holdout of "
        f"**{v['holdout_days']:.0f} days** to {v['holdout_end']}, "
        f"{v['customers']:,} customers. BG/NBD (purchases) + Gamma-Gamma (spend), MAP fit."
    )
    rows = []
    for target, models in [("orders", ["bgnbd", "naive"]), ("revenue", ["bgnbd_gg", "naive"])]:
        for mname in models:
            mm = v[target][mname]
            rows.append(
                {
                    "target": target,
                    "model": mname,
                    "MAE": round(mm["MAE"], 2),
                    "RMSE": round(mm["RMSE"], 2),
                    "total error %": round(mm["total_error_pct"], 1),
                    "Spearman": round(mm["spearman"], 3),
                }
            )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    r = v["revenue"]
    st.write(
        f"Top 20% by predicted value earned **{r['top20_share_bgnbd_gg']:.1%}** of actual "
        f"holdout revenue (naive ranking {r['top20_share_naive']:.1%})."
    )
    st.info(
        "Revenue is under-predicted mainly for the biggest customers in the Q4 peak: the "
        "models assume a steady buying rate. Use CLV to rank customers, not as a budget."
    )


def page_models() -> None:
    st.header("Recommender comparison (test window Sep–Nov 2011)")
    res = get("/models/comparison")
    for protocol, title in [
        ("all", "All purchases count (repeats included)"),
        ("new", "Only new-to-customer items (discovery)"),
    ]:
        df = pd.DataFrame(res[protocol])
        st.subheader(title)
        st.altair_chart(bar(df, "ndcg", "model", "NDCG@10", ".3f"), width="stretch")
        cols = [
            "model",
            "precision",
            "recall",
            "ndcg",
            "map",
            "hit_rate",
            "coverage",
            "diversity",
            "pop_bias",
            "ndcg 95% CI",
        ]
        st.dataframe(df[cols].round(4), hide_index=True, width="stretch")


def page_assistant() -> None:
    st.header("Assistant")
    st.caption(
        "Answers come only from tool results (profiles, recommendations, segments, "
        "read-only SQL). Out-of-scope questions are refused."
    )
    hist = st.session_state.setdefault("chat", [])
    for msg in hist:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for tc in msg.get("tools", []):
                with st.expander(f"🔧 {tc['name']}({tc['args']})"):
                    st.code(tc["result"][:3000], language="json")
    if prompt := st.chat_input("e.g. What segment is customer 14156 in?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Thinking…"):
            past = [{"role": m["role"], "content": m["content"]} for m in hist[-6:]]
            out = post("/assistant/chat", {"message": prompt, "history": past})
            st.markdown(out["answer"])
            for tc in out["tool_calls"]:
                with st.expander(f"🔧 {tc['name']}({tc['args']})"):
                    st.code(tc["result"][:3000], language="json")
        hist.append({"role": "user", "content": prompt})
        hist.append({"role": "assistant", "content": out["answer"], "tools": out["tool_calls"]})


PAGES = {
    "Customer lookup": page_customer,
    "Segments": page_segments,
    "CLV": page_clv,
    "Model comparison": page_models,
    "Assistant": page_assistant,
}
choice = st.sidebar.radio("ShopSage", list(PAGES))
st.sidebar.caption(f"API: {API_URL}")
PAGES[choice]()
