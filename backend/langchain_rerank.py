from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any

from backend.rerank_utils import (
    apply_llm_rerank_result,
    mark_llm_rerank_failed,
    normalize_llm_rerank_output,
)
from utils.paper import ArxivPaper

PROMPT_TEMPLATE = (
    "Project overview:\n"
    "{overview}\n\n"
    "Candidate paper:\n"
    "Title: {title}\n"
    "Abstract: {abstract}"
)

SYSTEM_PROMPT = """ROLE: You are a Literature Screening Agent.

TASK:
Given a project overview and a candidate paper (title + abstract), decide whether the paper is relevant to the project.

TOOLS:
You may call search_api(query) ONLY when one of the following is true:

(1) You encounter ONE key term / acronym / benchmark / method name
    that you do not understand, AND
    not understanding it materially affects relevance judgment.

(2) The abstract explicitly mentions a task, benchmark, dataset, or method
    WITHOUT providing any explanation or context,
    and understanding what it is (at a high level) is necessary
    to decide relevance.

(3) The abstract is missing/empty. In this case, you MAY use search_api
    to retrieve the paper's abstract or an official summary.

Tool use constraints:
- If using (3), search is ONLY to retrieve the abstract/official summary from primary sources.
  Prefer: arXiv, OpenReview, ACL Anthology, publisher/venue official page, authors' project page.
  Avoid relying on blogs or third-party interpretations when possible.
- Do NOT use search to invent results, claims, or novelty not supported by the retrieved abstract/summary.
- The search query must be SHORT (2–12 words).
  For (3) prefer: "<paper title> abstract" or "<paper title> arXiv".
- Use at most ONE search call per paper unless absolutely necessary.


RULES:
1) Start with Overview, Title, Abstract.
2) If ambiguity blocks decision, call the tool with a short query about the ambiguous term.
3) If key info is missing from the abstract, do NOT guess. Use action="clarify" and ask up to 3 focused questions in reasons.
4) Evidence grounding:
   - Quote short phrases from title/abstract when giving reasons.
   - If you used search, include one reason prefixed exactly with: "From search: ...".
5) Output MUST be strict JSON only. No markdown, no extra keys, no trailing text.

SCORING (fit_score 0-10):
0-2 irrelevant; 3-4 weak; 5-6 partial; 7-8 strong; 9-10 near-perfect.

ACTION (choose exactly one):
"reject" | "maybe_read" | "shortlist" | "clarify"

OUTPUT FORMAT (STRICT CONTRACT):
Return ONLY valid JSON with EXACT keys:
- relevant (boolean)
- fit_score (number 0-10)
- reasons (list of 2-5 strings)
- action (string from the action set)
- used_search (boolean; true if tool was called at least once, otherwise false)

OUTPUT HARD CONSTRAINT:
- Your entire response MUST be exactly one JSON object and nothing else.
- Do NOT wrap it in ```json fences.
- Do NOT include any commentary before or after the JSON.
- The first character of your response must be '{' and the last character must be '}'.

SELF-CHECK BEFORE RESPONDING:
If your draft response contains ANY text outside the JSON object (including code fences),
delete it and output ONLY the JSON object.
"""


@contextmanager
def _with_local_no_proxy():
    keys = ("NO_PROXY", "no_proxy")
    originals = {key: os.environ.get(key) for key in keys}
    addrs = ["localhost", "127.0.0.1", "::1"]

    def _merge(value: str | None) -> str:
        if not value:
            return ",".join(addrs)
        parts = [item.strip() for item in value.split(",") if item.strip()]
        for addr in addrs:
            if addr not in parts:
                parts.append(addr)
        return ",".join(parts)

    try:
        for key in keys:
            os.environ[key] = _merge(os.environ.get(key))
        yield
    finally:
        for key, value in originals.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _build_search_tool():
    try:
        from langchain_community.utilities.searchapi import SearchApiAPIWrapper
    except ImportError as exc:
        raise ImportError(
            "langchain-community is required for SearchApi. Install with `pip install langchain-community`."
        ) from exc
    from langchain_core.tools import Tool

    api_key = os.environ.get("SEARCHAPI_API_KEY") or os.environ.get("SEARCHAPI_KEY") or ""
    engine = os.environ.get("SEARCHAPI_ENGINE", "google")
    max_results = _get_env_int("SEARCHAPI_MAX_RESULTS", 5)
    max_snippet_length = _get_env_int("SEARCHAPI_MAX_SNIPPET_LENGTH", 100)

    if not api_key:
        logging.warning("SEARCHAPI_API_KEY is not set; search_api tool will return empty results.")

        def search_api(query: str) -> str:
            return "SearchAPI key missing; no results."

        return Tool(
            name="search_api",
            func=search_api,
            description=(
                "Search for brief context on a term or to retrieve a paper abstract. "
                "Input should be a short query (2-12 words)."
            ),
        )

    wrapper = SearchApiAPIWrapper(engine=engine, searchapi_api_key=api_key)

    def search_api(query: str) -> str:
        try:
            results = wrapper.results(query=query)
        except Exception as exc:
            return f"SearchAPI error: {exc}"
        organic = results.get("organic_results", [])[:max_results]
        simplified = [
            {
                "title": (item.get("title") or "")[:max_snippet_length],
                "link": item.get("link") or "",
                "snippet": (item.get("snippet") or "")[:max_snippet_length],
            }
            for item in organic
        ]
        return json.dumps(simplified, ensure_ascii=True)

    return Tool(
        name="search_api",
        func=search_api,
        description=(
            "Search for brief context on a term or to retrieve a paper abstract. "
            "Input should be a short query (2-12 words)."
        ),
    )


def _build_llm(model: str, base_url: str, temperature: float):
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError(
            "langchain-ollama is required for the Ollama model. Install with `pip install langchain-ollama`."
        ) from exc

    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )


def _extract_tool_calls(message: Any) -> list[dict[str, Any]]:
    tool_calls = []
    if hasattr(message, "tool_calls") and message.tool_calls:
        tool_calls = list(message.tool_calls)
    elif hasattr(message, "additional_kwargs"):
        raw = message.additional_kwargs.get("tool_calls") if message.additional_kwargs else None
        if raw:
            tool_calls = list(raw)
    return [dict(tc) if not isinstance(tc, dict) else tc for tc in tool_calls]


def _tool_call_to_query(call: dict[str, Any]) -> tuple[str, str | None, str | None]:
    name = call.get("name")
    args = call.get("args")
    call_id = call.get("id")
    if not name and "function" in call:
        func = call.get("function") or {}
        name = func.get("name")
        call_id = call.get("id")
        raw_args = func.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {"query": raw_args}
    query = None
    if isinstance(args, dict):
        query = args.get("query") or next(iter(args.values()), None)
    elif isinstance(args, str):
        query = args
    return str(name or ""), str(query) if query is not None else None, str(call_id) if call_id else None


def _run_with_optional_tool(prompt_text: str, llm: Any, tool: Any) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    llm_with_tools = llm.bind_tools([tool])
    with _with_local_no_proxy():
        response = llm_with_tools.invoke(messages)
    tool_calls = _extract_tool_calls(response)
    if not tool_calls:
        return getattr(response, "content", "") or ""

    name, query, call_id = _tool_call_to_query(tool_calls[0])
    if not query or name != tool.name:
        return getattr(response, "content", "") or ""

    tool_output = tool.func(query)
    tool_message = ToolMessage(content=str(tool_output), tool_call_id=call_id or "")
    with _with_local_no_proxy():
        final = llm.invoke([*messages, response, tool_message])
    return getattr(final, "content", "") or ""


def langchain_llm_rerank(
    overview_text: str,
    papers: list[ArxivPaper],
    model: str = "qwen2.5:32b",
    base_url: str = "http://localhost:11434",
    temperature: float | None = None,
) -> list[ArxivPaper]:
    if temperature is None:
        temperature = _get_env_float("LANGCHAIN_TEMPERATURE", 0.1)

    llm = _build_llm(model=model, base_url=base_url, temperature=temperature)
    tool = _build_search_tool()

    for paper in papers:
        try:
            prompt_text = PROMPT_TEMPLATE.format(
                overview=overview_text,
                title=paper.title,
                abstract=paper.summary,
            )
            output = _run_with_optional_tool(prompt_text, llm, tool)
            data = _extract_json(output)
            normalized = normalize_llm_rerank_output(data)
            apply_llm_rerank_result(paper, normalized)
        except Exception as exc:
            logging.warning("LLM rerank failed for %s: %s", paper.arxiv_id, exc)
            mark_llm_rerank_failed(paper)
    return papers
