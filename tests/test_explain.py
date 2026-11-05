import pytest

from shopsage.llm.explain import (
    Explainer,
    ProductMatcher,
    _clean,
    check_explanation,
    evidence_facts,
    template_explanation,
)

CATALOGUE = [
    "WHITE HANGING HEART T-LIGHT HOLDER",
    "RED HANGING HEART T-LIGHT HOLDER",
    "LUNCH BAG WOODLAND",
    "ROUND SNACK BOXES SET OF4 WOODLAND",
    "JUMBO BAG RED RETROSPOT",
]
ITEM = {
    "stock_code": "22328",
    "description": "ROUND SNACK BOXES SET OF4 WOODLAND",
    "evidence": {
        "popularity": {"buyers": 812, "rank": 20, "catalog_size": 4611},
        "similar_past_purchases": [
            {
                "stock_code": "20726",
                "description": "LUNCH BAG WOODLAND",
                "co_purchase_customers": 301,
                "similarity": 0.41,
            }
        ],
    },
}


@pytest.fixture
def matcher():
    return ProductMatcher(CATALOGUE)


def test_faithful_text_passes(matcher):
    text = "Because you bought the Lunch Bag Woodland: 301 customers bought both."
    assert check_explanation(text, ITEM, matcher).ok


def test_rejects_product_not_in_evidence(matcher):
    text = "You may also like the White Hanging Heart T-Light Holder."
    res = check_explanation(text, ITEM, matcher)
    assert not res.ok and "not in evidence" in res.reason


def test_rejects_invented_number(matcher):
    res = check_explanation("Bought by 999 customers.", ITEM, matcher)
    assert not res.ok and "999" in res.reason


def test_rejects_claims_about_tastes(matcher):
    res = check_explanation("Since you enjoy woodland designs, try this.", ITEM, matcher)
    assert not res.ok and "tastes" in res.reason


def test_date_rendering_without_leading_zero_is_allowed(matcher):
    item = {
        **ITEM,
        "evidence": {
            **ITEM["evidence"],
            "your_history": {"times_ordered": 3, "last_ordered": "2011-12-07"},
        },
    }
    assert check_explanation("You last ordered it on 7 December 2011.", item, matcher).ok


def test_clean_strips_numbering_and_limits_sentences():
    raw = "1. First sentence. 2. Second one! Third sentence."
    assert _clean(raw) == "First sentence. Second one!"


def test_template_uses_only_evidence():
    t = template_explanation(ITEM)
    assert "301" in t and "Lunch Bag Woodland" in t
    assert check_explanation(t, ITEM, ProductMatcher(CATALOGUE)).ok


def test_evidence_facts_mention_all_numbers():
    facts = " ".join(evidence_facts(ITEM))
    assert "812" in facts and "301" in facts


def test_explainer_falls_back_to_template_and_caches(tmp_path):
    calls = []

    def bad_llm(messages):
        calls.append(messages)
        return "You will love the White Hanging Heart T-Light Holder."

    ex = Explainer(CATALOGUE, llm=bad_llm, cache_path=tmp_path / "c.sqlite")
    first = ex.explain(ITEM)
    assert first["source"] == "template" and first["check"].startswith("rejected")
    second = ex.explain(ITEM)
    assert second["cached"] and len(calls) == 1


def test_explainer_survives_llm_errors():
    def broken(messages):
        raise TimeoutError

    out = Explainer(CATALOGUE, llm=broken, cache_path=None).explain(ITEM)
    assert out["source"] == "template" and "llm error" in out["check"]


def test_explainer_accepts_good_llm_text():
    good = lambda m: "Because you bought the Lunch Bag Woodland: 301 customers bought both."  # noqa: E731
    out = Explainer(CATALOGUE, llm=good, cache_path=None).explain(ITEM)
    assert out["source"] == "llm"
