"""Turn a recommendation's evidence into 1-2 plain sentences.

The LLM sees only the evidence JSON. `check_explanation` rejects any text that mentions a
catalogue product (or a number) that isn't in the evidence; rejected or failed generations fall
back to a deterministic template. Results are cached in SQLite.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from shopsage.config import ARTIFACTS_DIR, get_settings

SYSTEM = (
    "You explain product recommendations for a UK gift-ware wholesaler. "
    "Write 1-2 short sentences addressed to the customer ('you') saying why the RECOMMENDED "
    "product fits them. Use ONLY the facts listed. Mention no products except the recommended "
    "one and those named in the facts. Any number you use must appear in the facts. Do not "
    "invent reasons, prices, discounts, ranks or trends. Never describe the customer's tastes, "
    "preferences, style or feelings, and don't praise the product. No lists, no numbering, "
    "no preamble."
)

STOPWORDS = {
    "SET",
    "OF",
    "THE",
    "AND",
    "WITH",
    "IN",
    "A",
    "FOR",
    "TO",
    "ON",
    "BOX",
    "PACK",
    "SMALL",
    "LARGE",
    "RED",
    "PINK",
    "BLUE",
    "WHITE",
    "GREEN",
    "BLACK",
    "DESIGN",
    "VINTAGE",
    "RETROSPOT",
    "YOU",
    "YOUR",
    "HAVE",
    "BOUGHT",
    "ALSO",
    "CUSTOMERS",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Z0-9']+", text.upper())


def _ngrams(words: list[str], n: int = 3) -> set[tuple[str, ...]]:
    grams = set()
    for i in range(len(words) - n + 1):
        g = tuple(words[i : i + n])
        if any(w not in STOPWORDS and len(w) >= 4 and not w.isdigit() for w in g):
            grams.add(g)
    return grams


def evidence_products(item: dict[str, Any]) -> list[str]:
    ev = item.get("evidence", {})
    names = [item["description"]]
    names += [p["description"] for p in ev.get("similar_past_purchases", [])]
    return names


def evidence_numbers(item: dict[str, Any]) -> set[str]:
    blob = json.dumps(item)
    nums = set(re.findall(r"\d+(?:\.\d+)?", blob))
    # same value, different rendering: "07" -> "7" (dates), "2.50" -> "2.5"
    nums |= {str(int(n)) for n in nums if n.isdigit()}
    return nums | {n.rstrip("0").rstrip(".") for n in nums if "." in n}


# claims about the customer's tastes or praise of the product: never in the evidence
SUBJECTIVE = re.compile(
    r"\b(you (?:enjoy|love|prefer|adore)|your (?:taste|style|preference|favou?rite|passion)|"
    r"preference for|perfect|charm|versatil|similar tastes|share your|you'll love|ideal)",
    re.IGNORECASE,
)


@dataclass
class Check:
    ok: bool
    reason: str = ""


class ProductMatcher:
    """Finds catalogue product names mentioned in free text (exact name or a distinctive 3-gram)."""

    def __init__(self, catalogue: Iterable[str]) -> None:
        self.names = sorted({c.upper() for c in catalogue if c}, key=len, reverse=True)
        self.gram_owner: dict[tuple[str, ...], set[str]] = {}
        for name in self.names:
            for g in _ngrams(_words(name)):
                self.gram_owner.setdefault(g, set()).add(name)

    def mentioned(self, text: str) -> set[str]:
        up = " ".join(_words(text))
        found = {n for n in self.names if len(n) > 8 and " ".join(_words(n)) in up}
        for g in _ngrams(_words(text)):
            found |= self.gram_owner.get(g, set())
        return found


def check_explanation(text: str, item: dict[str, Any], matcher: ProductMatcher) -> Check:
    if not text or len(text) > 600:
        return Check(False, "empty or too long")
    allowed = {n.upper() for n in evidence_products(item)}
    allowed_grams = set().union(*(_ngrams(_words(n)) for n in allowed))
    extra = set()
    for name in matcher.mentioned(text):
        if name in allowed:
            continue
        # a 3-gram shared with an allowed product is fine (e.g. "LUNCH BAG RED")
        shared = _ngrams(_words(name)) & _ngrams(_words(text))
        if shared and shared <= allowed_grams:
            continue
        extra.add(name)
    if extra:
        return Check(False, f"names products not in evidence: {sorted(extra)[:3]}")
    subj = SUBJECTIVE.search(text)
    if subj:
        return Check(False, f"unsupported claim about tastes or quality: '{subj.group(0)}'")
    nums = set(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))
    bad = {n for n in nums if n not in evidence_numbers(item)}
    if bad:
        return Check(False, f"numbers not in evidence: {sorted(bad)}")
    return Check(True)


def template_explanation(item: dict[str, Any]) -> str:
    ev = item.get("evidence", {})
    name = item["description"].title()
    hist = ev.get("your_history")
    if hist:
        return (
            f"You have ordered {name} {hist['times_ordered']} times, most recently on "
            f"{hist['last_ordered']}."
        )
    sims = ev.get("similar_past_purchases") or []
    if sims:
        s = sims[0]
        return (
            f"You bought {s['description'].title()}, and {s['co_purchase_customers']} customers "
            f"bought both it and {name}."
        )
    pop = ev.get("popularity", {})
    return f"{name} is one of our popular items, bought by {pop.get('buyers', 'many')} customers."


def evidence_facts(item: dict[str, Any]) -> list[str]:
    """The evidence as short plain-English facts (what the LLM is allowed to say)."""
    ev = item.get("evidence", {})
    name = item["description"]
    facts = []
    pop = ev.get("popularity")
    if pop:
        facts.append(f"{name} was bought by {pop['buyers']} customers.")
    for s in ev.get("similar_past_purchases", []):
        facts.append(
            f"You bought {s['description']}; {s['co_purchase_customers']} customers bought both "
            f"{s['description']} and {name}."
        )
    hist = ev.get("your_history")
    if hist:
        facts.append(
            f"You have ordered {name} {hist['times_ordered']} times; last on {hist['last_ordered']}."
        )
    return facts


EXAMPLE_FACTS = (
    "RECOMMENDED: ROUND SNACK BOXES SET OF4 WOODLAND\n"
    "- ROUND SNACK BOXES SET OF4 WOODLAND was bought by 812 customers.\n"
    "- You bought LUNCH BAG WOODLAND; 301 customers bought both LUNCH BAG WOODLAND and "
    "ROUND SNACK BOXES SET OF4 WOODLAND."
)
EXAMPLE_ANSWER = (
    "Because you bought the Lunch Bag Woodland, you might like the Round Snack Boxes Set Of4 "
    "Woodland: 301 customers bought both."
)


def _prompt(item: dict[str, Any]) -> list[dict[str, str]]:
    facts = "\n".join(f"- {f}" for f in evidence_facts(item))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": EXAMPLE_FACTS},
        {"role": "assistant", "content": EXAMPLE_ANSWER},
        {"role": "user", "content": f"RECOMMENDED: {item['description']}\n{facts}"},
    ]


def _clean(text: str) -> str:
    """Strip quotes and list numbering; keep at most two sentences."""
    text = re.sub(r"(?:^|(?<=[.!?:]))\s*(?:\d+[.)]|[-*])\s+", " ", text.strip().strip('"'))
    text = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:2])


class Explainer:
    def __init__(
        self,
        catalogue: Iterable[str],
        llm: Callable[[list[dict]], str] | None = None,
        cache_path: Path | None = ARTIFACTS_DIR / "explain_cache.sqlite",
    ) -> None:
        self.matcher = ProductMatcher(catalogue)
        self._llm = llm
        self.cache_path = cache_path

    @cached_property
    def _db(self) -> sqlite3.Connection | None:
        if self.cache_path is None:
            return None
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(str(self.cache_path), check_same_thread=False)
        db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        return db

    def llm(self, messages: list[dict]) -> str:
        if self._llm is not None:
            return self._llm(messages)
        from shopsage.llm.client import chat

        return chat(messages, max_tokens=120)

    @staticmethod
    def key(item: dict[str, Any]) -> str:
        payload = json.dumps(
            {"m": get_settings().llm_model, "d": item["description"], "e": item.get("evidence")},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def generate(self, item: dict[str, Any]) -> str:
        """Raw LLM output (no check), used by the faithfulness evaluation."""
        return _clean(self.llm(_prompt(item)))

    def explain(self, item: dict[str, Any]) -> dict[str, Any]:
        k = self.key(item)
        if self._db is not None:
            row = self._db.execute("SELECT value FROM cache WHERE key = ?", (k,)).fetchone()
            if row:
                return {**json.loads(row[0]), "cached": True}
        try:
            text = self.generate(item)
            check = check_explanation(text, item, self.matcher)
        except Exception as e:  # LLM down, timeout, ... -> template
            text, check = "", Check(False, f"llm error: {type(e).__name__}")
        if check.ok:
            out = {"text": text, "source": "llm", "check": "passed"}
        else:
            out = {
                "text": template_explanation(item),
                "source": "template",
                "check": f"rejected: {check.reason}",
                "rejected_text": text,
            }
        if self._db is not None:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, json.dumps(out)))
            self._db.commit()
        return {**out, "cached": False}
