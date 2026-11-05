"""LangGraph assistant: an LLM node that may call tools, a tool node, and a loop between them.

Small local models (e.g. Mistral 7B on Ollama) often write the tool call as JSON text instead of
a native tool call, or describe what they would do instead of doing it. Three model-agnostic
safeguards handle this:

- `parse_text_tool_calls` turns JSON tool calls written in the message text into real ones;
- if the first reply has no tool call and is not the refusal, the graph nudges once;
- `ground_answer` checks every number in the final answer against the tool results (and the
  question); an answer with unverifiable numbers is replaced by the raw tool output.
"""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from shopsage.agent.tools import TOOLS
from shopsage.config import get_settings

REFUSAL = "Sorry, I can only help with ShopSage's customers, products, sales and recommendations."
TOOL_NAMES = {f.__name__ for f in TOOLS}

SYSTEM_PROMPT = f"""You are ShopSage, an analytics assistant for a UK online gift-ware wholesaler
(data 2009-12-01 to 2011-12-09). You answer questions about its customers, products, sales,
recommendations, customer segments and customer lifetime value (CLV).

Rules:
1. Every number, product or customer fact in your answer must come from a tool result in this
   conversation. Never estimate or use outside knowledge. If a tool returns an error or no
   data, say so plainly.
2. For any data question, first call a tool. Use run_sql for counts, totals, rankings and dates;
   search_products to find a product's stock code from its name.
3. To call a tool, reply with ONLY a JSON list, for example:
   [{{"name": "get_customer_profile", "arguments": {{"customer_id": 12345}}}}]
4. After you get tool results, answer in 1-4 sentences and cite the tool in square brackets,
   e.g. "Customer 12345 is a Champion [get_customer_profile]."
5. If the question is not about this shop's data (general knowledge, coding, other companies,
   personal data such as emails or addresses, or requests to change data), reply exactly:
   "{REFUSAL}"

Tools: {", ".join(sorted(TOOL_NAMES))}."""

NO_TOOLS_PROMPT = f"""You are ShopSage, an analytics assistant for a UK online gift-ware
wholesaler (data 2009-12-01 to 2011-12-09). Answer questions about its customers, products,
sales, recommendations, customer segments and customer lifetime value in 1-4 sentences.
If the question is not about this shop's data (general knowledge, coding, other companies,
personal data, or requests to change data), reply exactly: "{REFUSAL}\""""

NUDGE = (
    "You did not call a tool. If this is a question about the shop's data, reply now with ONLY "
    'the JSON tool call, e.g. [{"name": "run_sql", "arguments": {"query": "SELECT ..."}}]. '
    f'Otherwise reply exactly: "{REFUSAL}"'
)


# Deterministic scope guard, applied before the LLM: requests to change data or for personal data
WRITE_INTENT = re.compile(
    r"\b(delete|drop|truncate|insert\s+into|alter\s+table|overwrite|remove)\b.{0,60}"
    r"\b(customers?|tables?|database|records?|rows?|data|orders?|products?)\b"
    r"|\b(update|modify|change|edit)\s+(the\s+)?(database|tables?|records?|rows?|data|"
    r"customer\s+\d+|prices?)\b",
    re.IGNORECASE,
)
PII = re.compile(
    r"\b(e-?mail|phone|telephone|home address|postal address|password|credit card|"
    r"bank account|date of birth)\b",
    re.IGNORECASE,
)


def out_of_scope_by_rule(message: str) -> bool:
    return bool(WRITE_INTENT.search(message) or PII.search(message))


def _tools() -> list[StructuredTool]:
    return [StructuredTool.from_function(f) for f in TOOLS]


def parse_text_tool_calls(content: str) -> list[dict[str, Any]]:
    """Find JSON tool calls ({"name": ..., "arguments": {...}}) written in plain text."""
    calls = []
    for start in [m.start() for m in re.finditer(r"[\[{]", content)]:
        for end in range(len(content), start, -1):
            if content[end - 1] not in "]}":
                continue
            try:
                obj = json.loads(content[start:end])
            except json.JSONDecodeError:
                continue
            items = obj if isinstance(obj, list) else [obj]
            for it in items:
                if isinstance(it, dict) and it.get("name") in TOOL_NAMES:
                    args = it.get("arguments", it.get("parameters", {}))
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    calls.append(
                        {
                            "name": it["name"],
                            "args": args if isinstance(args, dict) else {},
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call",
                        }
                    )
            if calls:
                return calls
            break
    return calls


def _dedupe(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for c in calls:
        key = (c["name"], json.dumps(c["args"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


NUM = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?")
LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")
# period names such as "12-month CLV" are labels, not data
PERIOD = re.compile(r"\b\d+[- ](?:month|day|week|year)s?\b", re.IGNORECASE)


def _numbers(text: str) -> set[float]:
    out = set()
    for n in NUM.findall(PERIOD.sub(" ", LIST_MARKER.sub(" ", text))):
        with contextlib.suppress(ValueError):
            out.add(float(n.replace(",", "")))
    return out


def ungrounded_numbers(answer: str, sources: list[str]) -> set[float]:
    """Numbers in the answer that don't match any number in the sources (to 2 decimals,
    or as a rounding of one: 133298.54 may appear as 133,299 or 133.3k is NOT accepted)."""
    allowed = set().union(*(_numbers(s) for s in sources)) if sources else set()
    bad = set()
    for x in _numbers(answer):
        if any(
            abs(x - a) < 0.006
            or abs(x - round(a)) < 1e-9
            or abs(x - round(a, 1)) < 1e-9
            or abs(x * 100 - a * 100) < 0.6
            or abs(x - a * 100) < 0.6
            for a in allowed
        ):
            continue
        bad.add(x)
    return bad


def ground_answer(answer: str, question: str, tool_results: list[str]) -> tuple[str, list[float]]:
    bad = ungrounded_numbers(answer, [question, *tool_results])
    if not bad:
        return answer, []
    if tool_results:
        raw = tool_results[-1]
        raw = raw if len(raw) < 1500 else raw[:1500] + "..."
        msg = (
            "I couldn't verify every number in my draft answer against the data, so here is "
            f"the raw result from the tool instead:\n{raw}"
        )
    else:
        msg = "I couldn't find data to answer that reliably."
    return msg, sorted(bad)


@lru_cache
def build_graph(use_tools: bool = True):
    s = get_settings()
    llm = ChatOpenAI(
        model=s.llm_model,
        base_url=s.openai_base_url,
        api_key=s.llm_api_key,  # type: ignore[arg-type]
        temperature=0,
        timeout=s.llm_timeout_s,
    )
    tools = _tools()
    bound = llm.bind_tools(tools) if use_tools else llm

    def agent(state: MessagesState) -> dict[str, list]:
        reply = bound.invoke(state["messages"])
        if use_tools and isinstance(reply, AIMessage):
            calls: list[dict[str, Any]] = [
                dict(c) for c in reply.tool_calls
            ] or parse_text_tool_calls(str(reply.content))
            if calls:
                reply = AIMessage(content="", tool_calls=_dedupe(calls)[:3])
        return {"messages": [reply]}

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        used_tool = any(isinstance(m, ToolMessage) for m in state["messages"])
        nudged = any(isinstance(m, HumanMessage) and m.content == NUDGE for m in state["messages"])
        if not used_tool and not nudged and REFUSAL not in str(last.content):
            return "nudge"
        return END

    def nudge(state: MessagesState) -> dict[str, list]:
        return {"messages": [HumanMessage(NUDGE)]}

    g = StateGraph(MessagesState)
    g.add_node("agent", agent)
    g.add_edge(START, "agent")
    if use_tools:
        g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
        g.add_node("nudge", nudge)
        g.add_conditional_edges("agent", route, {"tools": "tools", "nudge": "nudge", END: END})
        g.add_edge("tools", "agent")
        g.add_edge("nudge", "agent")
    else:
        g.add_edge("agent", END)
    return g.compile()


def run_agent(
    message: str, history: list[dict[str, str]] | None = None, use_tools: bool = True
) -> dict[str, Any]:
    """Answer one user message. Returns the answer and every tool call with its result."""
    if use_tools and out_of_scope_by_rule(message):
        return {
            "answer": REFUSAL,
            "draft": REFUSAL,
            "tool_calls": [],
            "ungrounded_numbers": [],
            "guard": "scope_rule",
        }
    msgs: list[Any] = [SystemMessage(SYSTEM_PROMPT if use_tools else NO_TOOLS_PROMPT)]
    for h in history or []:
        cls = HumanMessage if h.get("role") == "user" else AIMessage
        msgs.append(cls(h.get("content", "")))
    msgs.append(HumanMessage(message))
    try:
        state = build_graph(use_tools).invoke({"messages": msgs}, {"recursion_limit": 14})
    except Exception as e:  # LLM down or recursion limit -> explicit failure, no invented answer
        return {
            "answer": f"The assistant failed ({type(e).__name__}); please try again.",
            "draft": "",
            "tool_calls": [],
            "ungrounded_numbers": [],
            "error": str(e)[:300],
        }
    results = {
        m.tool_call_id: str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)
    }
    calls = [
        {"name": tc["name"], "args": tc["args"], "result": results.get(tc["id"] or "", "")}
        for m in state["messages"]
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
    ]
    content = state["messages"][-1].content
    draft = (content if isinstance(content, str) else json.dumps(content)).strip()
    answer, bad = (
        ground_answer(draft, message, [str(c["result"]) for c in calls])
        if use_tools
        else (draft, [])
    )
    return {"answer": answer, "draft": draft, "tool_calls": calls, "ungrounded_numbers": bad}
