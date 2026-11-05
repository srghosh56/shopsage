import asyncio
import json

import pytest

from shopsage.agent import graph
from shopsage.agent.evaluate import Question, is_refusal, number_present, score
from shopsage.agent.sql import UnsafeQuery, run_readonly, validate

SAFE = [
    "SELECT count(*) FROM customers",
    "WITH t AS (SELECT * FROM customers) SELECT country, count(*) FROM t GROUP BY 1",
    "SELECT 1 UNION SELECT 2",
]
UNSAFE = [
    "DELETE FROM customers",
    "DROP TABLE customers",
    "SELECT 1; DROP TABLE customers",
    "INSERT INTO customers VALUES (1)",
    "UPDATE customers SET country = 'x'",
    "PRAGMA database_list",
    "COPY customers TO 'x.csv'",
    "ATTACH 'other.db'",
    "SELECT * INTO t2 FROM customers",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM 'data/raw/online_retail_ii.parquet'",
    "SELECT * FROM duckdb_settings()",
    "SELECT getenv('HOME')",
    "SELECT * FROM information_schema.tables",
    "SELECT * FROM main.customers",
    "CREATE TABLE z AS SELECT 1",
    "not sql at all (",
]


@pytest.mark.parametrize("q", SAFE)
def test_sql_allows_selects_and_adds_limit(q):
    assert "LIMIT 100" in validate(q)


@pytest.mark.parametrize("q", UNSAFE)
def test_sql_blocks_unsafe(q):
    with pytest.raises(UnsafeQuery):
        validate(q)


def test_sql_caps_large_limits_keeps_small():
    assert "LIMIT 100" in validate("SELECT * FROM customers LIMIT 100000")
    assert "LIMIT 5" in validate("SELECT * FROM customers LIMIT 5")


def test_sql_runs_read_only(artifacts):
    from shopsage import config

    out = run_readonly("SELECT count(*) AS n FROM customers", config.DUCKDB_PATH)
    assert out["columns"] == ["n"] and out["rows"][0][0] > 0


def test_parse_text_tool_calls():
    calls = graph.parse_text_tool_calls(
        ' [{"name":"get_customer_profile","arguments":{"customer_id": 12345}}]\n\nTo find...'
    )
    assert calls[0]["name"] == "get_customer_profile"
    assert calls[0]["args"] == {"customer_id": 12345}
    assert graph.parse_text_tool_calls('{"name": "unknown_tool", "arguments": {}}') == []
    assert graph.parse_text_tool_calls("no json here") == []


def test_ungrounded_numbers():
    src = ['{"clv_12m_gbp": 133298.54, "orders": 143, "share": 0.1685}']
    assert graph.ungrounded_numbers("CLV is £133,298.54 from 143 orders.", src) == set()
    assert graph.ungrounded_numbers("About 16.85% of revenue.", src) == set()
    assert graph.ungrounded_numbers("1. Item one\n2. Item two (143 orders)", src) == set()
    assert graph.ungrounded_numbers("CLV is £150,000.", src) == {150000.0}
    assert graph.ungrounded_numbers("Their 12-month CLV is £133,298.54.", src) == set()


def test_ground_answer_replaces_unverified_drafts():
    ans, bad = graph.ground_answer("It is 999.", "q", ['{"n": 5}'])
    assert bad == [999.0] and '{"n": 5}' in ans
    ans, bad = graph.ground_answer("It is 5.", "q", ['{"n": 5}'])
    assert ans == "It is 5." and bad == []


@pytest.mark.parametrize(
    ("msg", "blocked"),
    [
        ("Delete customer 123 from the database.", True),
        ("DROP TABLE customers", True),
        ("What is the email address of customer 123?", True),
        ("Change customer 123 country to France", True),
        ("Update me on the top products", False),
        ("How many orders were returned?", False),
        ("Which products did customer 123 buy?", False),
    ],
)
def test_scope_rule(msg, blocked):
    assert graph.out_of_scope_by_rule(msg) is blocked


def test_run_agent_refuses_by_rule_without_llm():
    out = graph.run_agent("Delete customer 1 from the database")
    assert out["answer"] == graph.REFUSAL and out["tool_calls"] == []


def test_tools_return_json(service, monkeypatch):
    from shopsage.agent import tools

    monkeypatch.setattr(tools, "get_service", lambda: service)
    cid = service.sample_customers(1)[0]
    prof = json.loads(tools.get_customer_profile(cid))
    assert prof["customer_id"] == cid
    recs = json.loads(tools.get_recommendations(cid, 3))
    assert len(recs["items"]) == 3
    assert "error" in json.loads(tools.get_customer_profile(1))
    assert "error" in json.loads(tools.get_segment_summary("VIP Gold"))


def test_mcp_server_exposes_all_tools():
    from shopsage.agent.mcp_server import build_server
    from shopsage.agent.tools import TOOLS

    names = {t.name for t in asyncio.run(build_server().list_tools())}
    assert names == {f.__name__ for f in TOOLS}


def test_eval_scoring_helpers():
    assert number_present(0.1685, "about 16.9% of revenue")
    assert number_present(1153, "There are 1,153 Champions.")
    assert not number_present(1153, "There are 1,000 Champions.")
    assert is_refusal(graph.REFUSAL)
    q = Question(1, "sql", "How many?", ["run_sql"], [5.0])
    res = {
        "answer": "5 [run_sql]",
        "draft": "5 [run_sql]",
        "tool_calls": [{"name": "run_sql", "result": '{"rows": [[5]]}'}],
    }
    s = score(q, res, with_tools=True)
    assert s["correct"] and s["tool_ok"] and s["draft_grounded"]
