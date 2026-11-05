"""Faithfulness evaluation of LLM explanations.

Generates 30 explanations (10 customers x 3 items, half 'discover', half 'buy again'), runs the
rule-based check on each, and asks an LLM judge whether every claim is supported by the facts.
Writes docs/labels/explanations.csv (with an empty `my_label` column for a human) and the
EXPLAIN section of docs/RESULTS.md. Existing human labels are never overwritten.
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from shopsage.config import DOCS_DIR, SEED, get_settings
from shopsage.llm.client import chat
from shopsage.llm.explain import Explainer, check_explanation, evidence_facts
from shopsage.reporting import md_table, write_section
from shopsage.service import get_service

log = logging.getLogger(__name__)
LABELS_CSV = DOCS_DIR / "labels" / "explanations.csv"
N_CUSTOMERS, PER_CUSTOMER = 10, 3

JUDGE_SYSTEM = (
    "You are a strict fact-checker. Check the EXPLANATION against the FACTS, claim by claim. "
    "Mark it UNFAITHFUL if ANY of these happen: (1) it names a product that is not in the facts "
    "or garbles a product name; (2) a number is wrong or attached to the wrong product pair; "
    "(3) it states the customer's feelings, tastes, preferences, hobbies or enjoyment (e.g. 'you "
    "enjoy', 'your preference for', 'fits your style') - the facts never say this; (4) it adds "
    "reasons or qualities not in the facts (e.g. 'perfect complement', 'charm', 'similar "
    "tastes'); (5) it claims a date, count or history not in the facts. Otherwise it is "
    "FAITHFUL; wording may differ. First list the claims briefly, then end with JSON on the "
    'last line: {"faithful": true or false, "reason": "<one short sentence>"}'
)


def judge(facts: list[str], explanation: str, model: str | None = None) -> tuple[int | None, str]:
    msg = "FACTS:\n" + "\n".join(f"- {f}" for f in facts) + f"\n\nEXPLANATION:\n{explanation}"
    raw = chat(
        [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": msg}],
        max_tokens=500,
        model=model,
    )
    found = re.findall(r"\{[^{}]*\"faithful\"[^{}]*\}", raw)
    try:
        obj = json.loads(found[-1])
        return int(bool(obj["faithful"])), str(obj.get("reason", ""))
    except (IndexError, json.JSONDecodeError, KeyError, TypeError):
        return None, f"unparseable: {raw[-200:]}"


def judge_col(model: str) -> str:
    return f"judge[{model}]"


def sample_items(n_customers: int = N_CUSTOMERS, per_customer: int = PER_CUSTOMER):
    s = get_service()
    rng = np.random.default_rng(SEED)
    eligible = [
        int(c) for c in s.segments.index if s.data.binary[s.data.user_index[int(c)]].nnz >= 5
    ]
    customers = rng.choice(sorted(eligible), size=n_customers, replace=False)
    for j, cid in enumerate(customers):
        mode = "discover" if j % 2 == 0 else "buy_again"
        for item in s.recommend(int(cid), per_customer, mode)["items"]:
            yield int(cid), mode, item


def run() -> pd.DataFrame:
    s = get_service()
    ex = Explainer(s.products.description.tolist(), cache_path=None)
    rows = []
    for i, (cid, mode, item) in enumerate(sample_items(), 1):
        text = ex.generate(item)
        check = check_explanation(text, item, ex.matcher)
        facts = evidence_facts(item)
        verdicts = {m: judge(facts, text, model=m) for m in get_settings().judges}
        rows.append(
            {
                "id": i,
                "customer_id": cid,
                "mode": mode,
                "stock_code": item["stock_code"],
                "recommended": item["description"],
                "facts": " | ".join(facts),
                "explanation": text,
                "check": "passed" if check.ok else f"rejected: {check.reason}",
                **{judge_col(m): "" if v[0] is None else v[0] for m, v in verdicts.items()},
                **{f"{judge_col(m)} reason": v[1] for m, v in verdicts.items()},
                "my_label": "",
            }
        )
        log.info("%d/30 check=%s judges=%s", i, check.ok, {m: v[0] for m, v in verdicts.items()})
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, col: str) -> dict[str, float]:
    judged = df[df[col].astype(str) != ""].copy()
    jf = judged[col].astype(int) == 1
    passed = judged.check == "passed"
    return {
        "n_judged": float(len(judged)),
        "judge_faithful_raw": float(jf.mean()),
        "judge_faithful_given_passed": float(jf[passed].mean()) if passed.any() else float("nan"),
        "judge_faithful_given_rejected": float(jf[~passed].mean())
        if (~passed).any()
        else float("nan"),
        # served = LLM text if it passed the check, otherwise the template, which only restates
        # the evidence and is counted as faithful
        "served_faithful": float(((passed & jf) | ~passed).mean()),
        "check_judge_agreement": float((passed == jf).mean()),
    }


def _pct(v: float) -> str:
    return "n/a" if v != v else f"{v:.0%}"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out = LABELS_CSV
    if LABELS_CSV.exists():
        old = pd.read_csv(LABELS_CSV, dtype=str, keep_default_na=False)
        if (old.get("my_label", pd.Series(dtype=str)).str.strip() != "").any():
            out = LABELS_CSV.with_name("explanations.new.csv")
            log.warning("human labels found in %s; writing new run to %s", LABELS_CSV, out)
    df = run()
    df.to_csv(out, index=False)
    judges = get_settings().judges
    stats = {m: summarise(df, judge_col(m)) for m in judges}
    pass_rate = float((df.check == "passed").mean())
    reasons = df.check[df.check != "passed"].str.split(":").str[1].str.strip().str[:40]
    rows = [
        {
            "metric": "passed the rule-based check (else template)",
            **{m: _pct(pass_rate) for m in judges},
        }
    ]
    for key, label in [
        ("judge_faithful_raw", "judge: raw LLM explanation faithful"),
        ("judge_faithful_given_passed", "judge: faithful among check-passed"),
        ("judge_faithful_given_rejected", "judge: faithful among check-rejected"),
        ("served_faithful", "served output faithful (passed & faithful, or template)"),
        ("check_judge_agreement", "rule check and judge agree"),
    ]:
        rows.append({"metric": label, **{m: _pct(stats[m][key]) for m in judges}})
    if len(judges) > 1:
        a, b = (df[judge_col(m)].astype(str) for m in judges[:2])
        both = (a != "") & (b != "")
        rows.append(
            {
                "metric": f"judge-vs-judge agreement ({judges[0]} vs {judges[1]})",
                judges[0]: _pct(float((a[both] == b[both]).mean())),
                judges[1]: "",
            }
        )
    md = f"""
Generator: `{get_settings().llm_model}` (from `.env`), prompted with the evidence as plain facts.
Judges: {", ".join(f"`{m}`" for m in judges)}, each with a strict claim-by-claim rubric, seeing
only the facts and the explanation. 10 seeded-random customers x 3 recommendations
(5 customers "discover", 5 "buy again") = {len(df)} explanations. Rule-check rejection reasons:
{ {k: int(v) for k, v in reasons.value_counts().items()} }. Generated by `python -m shopsage.llm.judge`; rows are in
`docs/labels/explanations.csv` for human labelling (`make faithfulness`).

{md_table(rows, ["metric", *judges])}
"""
    write_section("EXPLAIN", md)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
