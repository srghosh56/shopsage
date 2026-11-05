"""Agreement between human labels (`my_label` in docs/labels/explanations.csv) and each LLM judge.

Label each row 1 (faithful) or 0 (not faithful), then run `make faithfulness`.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from shopsage.llm.judge import LABELS_CSV
from shopsage.reporting import write_section

TRUE = {"1", "yes", "y", "true", "faithful"}
FALSE = {"0", "no", "n", "false", "unfaithful"}


def parse(v: str) -> int | None:
    v = str(v).strip().lower()
    return 1 if v in TRUE else 0 if v in FALSE else None


def agreement(df: pd.DataFrame, col: str) -> dict[str, float] | None:
    h = df.my_label.map(parse)
    j = df[col].map(parse)
    mask = h.notna() & j.notna()
    if not mask.any():
        return None
    hh, jj = h[mask].astype(int), j[mask].astype(int)
    kappa = cohen_kappa_score(hh, jj) if len(set(hh) | set(jj)) > 1 else float("nan")
    return {
        "n_labelled": float(mask.sum()),
        "agreement": float((hh == jj).mean()),
        "cohen_kappa": float(kappa),
        "human_faithful_rate": float(hh.mean()),
        "judge_faithful_rate": float(jj.mean()),
    }


def guard_summary(df: pd.DataFrame) -> str:
    """Compare human labels with the rule check and split them by recommendation mode."""
    h = df.my_label.map(parse)
    labelled = df[h.notna()].assign(human=h[h.notna()].astype(int))
    passed = labelled.check.eq("passed")
    rows = [
        ("passed the rule check", labelled[passed]),
        ("rejected by the rule check", labelled[~passed]),
        *((f"mode: {mode}", g) for mode, g in labelled.groupby("mode")),
    ]
    lines = ["| group | labelled | human faithful |", "|---|---:|---:|"]
    for name, group in rows:
        if len(group):
            lines.append(f"| {name} | {len(group)} | {group.human.sum()} |")
    return "\n".join(lines)


def main() -> None:
    if not LABELS_CSV.exists():
        print(f"{LABELS_CSV} not found; run `make explain-eval` first.")
        return
    df = pd.read_csv(LABELS_CSV, dtype=str, keep_default_na=False)
    cols = [c for c in df.columns if c.startswith("judge[") and not c.endswith("reason")]
    results = {c: agreement(df, c) for c in cols}
    if all(r is None for r in results.values()):
        print(f"No human labels yet. Fill my_label (1 = faithful, 0 = not) in {LABELS_CSV}.")
        return
    lines = [
        "| judge | labelled | agreement | Cohen's kappa | human faithful | judge faithful |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for c, r in results.items():
        if r is not None:
            lines.append(
                f"| `{c[6:-1]}` | {r['n_labelled']:.0f} | {r['agreement']:.0%} | "
                f"{r['cohen_kappa']:.2f} | {r['human_faithful_rate']:.0%} | "
                f"{r['judge_faithful_rate']:.0%} |"
            )
    md = "Human labels vs each LLM judge (`make faithfulness`):\n\n" + "\n".join(lines)
    md += "\n\n" + guard_summary(df)
    write_section("HUMAN_AGREEMENT", md)
    print(md)


if __name__ == "__main__":
    main()
