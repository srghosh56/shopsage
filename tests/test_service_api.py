import pytest
from fastapi.testclient import TestClient

from shopsage.llm.explain import Explainer
from shopsage.search.products import ProductSearch, rrf


@pytest.fixture(scope="module")
def client(service, tmp_path_factory):
    from shopsage.api import main

    main.get_service.cache_clear()
    main.app.dependency_overrides[main.svc] = lambda: service
    fake = Explainer(
        service.products.description.tolist(),
        llm=lambda m: "Popular with other customers.",
        cache_path=tmp_path_factory.mktemp("cache") / "c.sqlite",
    )
    main.get_explainer.cache_clear()
    original = main.get_explainer
    main.get_explainer = lambda: fake  # type: ignore[assignment]
    yield TestClient(main.app)
    main.get_explainer = original
    main.app.dependency_overrides.clear()


@pytest.fixture(scope="module")
def cid(service):
    return service.sample_customers(5)[0]


def test_profile(service, cid):
    p = service.customer_profile(cid)
    assert p["segment"] and p["clv"]["clv_12m_gbp"] >= 0
    assert 0 <= p["clv"]["p_alive"] <= 1
    assert p["rfm"]["orders"] >= 1


def test_discover_excludes_bought_items(service, cid):
    recs = service.recommend(cid, 5, "discover")
    bought = set(service.data.item_ids[service.data.binary[service.data.user_index[cid]].indices])
    assert recs["model"] == "als"
    assert not {r["stock_code"] for r in recs["items"]} & bought


def test_buy_again_has_history_evidence(service, cid):
    recs = service.recommend(cid, 3, "buy_again")
    assert all("your_history" in r["evidence"] for r in recs["items"])


def test_evidence_co_purchase_counts_are_real(service, cid):
    item = service.recommend(cid, 1)["items"][0]
    i = service.data.item_index[item["stock_code"]]
    for s in item["evidence"]["similar_past_purchases"]:
        j = service.data.item_index[s["stock_code"]]
        b = service.data.binary
        both = int((b[:, i].multiply(b[:, j])).sum())
        assert s["co_purchase_customers"] == both


def test_cold_start_uses_popularity(service):
    assert service.recommend(1, 3)["model"] == "popularity"


def test_similar_items(service):
    code = str(service.data.item_ids[0])
    out = service.similar_items(code, 3)
    assert len(out["similar"]) == 3
    assert code not in {s["stock_code"] for s in out["similar"]}


def test_api_endpoints(client, cid):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get(f"/customers/{cid}").status_code == 200
    assert client.get("/customers/1").status_code == 404
    r = client.get(f"/customers/{cid}/recommendations", params={"k": 3})
    assert r.status_code == 200 and len(r.json()["items"]) == 3
    assert client.get(f"/customers/{cid}/recommendations", params={"mode": "x"}).status_code == 422
    r = client.get(f"/customers/{cid}/explanations", params={"k": 2})
    assert all(i["explanation"]["text"] for i in r.json()["items"])
    assert client.get("/segments").json()["segments"]
    assert client.get("/items/NOPE/similar").status_code == 404


def test_rrf():
    scores = rrf([[1, 2, 3], [3, 1]], k=60)
    assert max(scores, key=scores.get) == 1
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)


def test_bm25_search(service):
    s = ProductSearch(service.products.reset_index(), use_embeddings=False, cache=None)
    word = service.products.description.iloc[0].split()[0]
    hits = s.search(word, 5, mode="bm25")
    assert hits and word.lower() in hits[0]["description"].lower()
