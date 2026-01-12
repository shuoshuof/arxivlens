from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json, os
import logging

from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy


from langchain_openai import ChatOpenAI
from langchain_community.utilities.searchapi import SearchApiAPIWrapper

from utils.paper import ArxivPaper
from backend.rerank_utils import (
    apply_llm_rerank_result,
    mark_llm_rerank_failed,
    normalize_llm_rerank_output,
)



def _build_langchain_agent(cfg):

    @dataclass
    class ResponseFormat:
        relevant: bool
        fit_score: float
        reasons: list[str]
        action: str
        used_search: bool

    def _build_llm(llm_cfg):
        # TODO: add support for other LLMs if needed, now only deepseek-chat supported
        if not llm_cfg.get("api_key"):
            raise ValueError("Missing LLM API key. Set LANGCHAIN_RERANK_LLM_API_KEY in .env.")
        model = ChatOpenAI(**llm_cfg)
        return model
    
    def _build_search_api_tool(tool_cfg: dict):
        if not tool_cfg.get("api_key"):
            raise ValueError("Missing search API key. Set LANGCHAIN_RERANK_SEARCH_API_KEY in .env.")
        @tool
        def search_api(query: str) -> str:
            """Search for brief context on a term or retrieve a paper abstract. 2-12 words."""
            wrapper = SearchApiAPIWrapper(
                api_key=tool_cfg["api_key"],
                engine=tool_cfg["engine"],
            )
            results = wrapper.search(
                query,
                max_results=tool_cfg["max_results"],
                max_snippet_length=tool_cfg["max_snippet_length"],
            )
            return json.dumps(results, ensure_ascii=False)

        return search_api

    def build_tool(tool_name: str, tool_cfg: dict):
        try:
            return {
                "search_api": _build_search_api_tool,
            }[tool_name](tool_cfg)
        except KeyError as exc:
            raise ValueError(f"Unknown tool: {tool_name}") from exc

    return create_agent(
        model=_build_llm(cfg["llm"]),
        tools=[build_tool(name, cfg["tools"][name]) for name in cfg.get("tools", {})],
        system_prompt=cfg["prompt"]["system"],
        response_format=ToolStrategy(ResponseFormat),
    )

def _struct_resp2dict(struct_resp) -> dict[str, Any]:
    sr_dict = asdict(struct_resp)
    return sr_dict

def langchain_llm_rerank(
    overview_text: str,
    papers: list[ArxivPaper],
    cfg_path: str = "data/langchain_rerank.json",
    **kwargs
) -> list[ArxivPaper]:

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert type(cfg) is dict
    prompt_cfg = cfg.get("prompt", {})
    if "system_path" in prompt_cfg:
        system_path = prompt_cfg["system_path"]
        if not os.path.isabs(system_path) and not os.path.exists(system_path):
            system_path = os.path.join(os.path.dirname(cfg_path), system_path)
        with open(system_path, "r", encoding="utf-8") as f:
            prompt_cfg["system"] = f.read().strip()
    if "template_path" in prompt_cfg:
        template_path = prompt_cfg["template_path"]
        if not os.path.isabs(template_path) and not os.path.exists(template_path):
            template_path = os.path.join(os.path.dirname(cfg_path), template_path)
        with open(template_path, "r", encoding="utf-8") as f:
            prompt_cfg["template"] = f.read().strip()
    cfg["prompt"] = prompt_cfg

    llm_key = os.environ.get("LANGCHAIN_RERANK_LLM_API_KEY")
    if llm_key:
        cfg.setdefault("llm", {})["api_key"] = llm_key
    tool_key = os.environ.get("LANGCHAIN_RERANK_SEARCH_API_KEY")
    if tool_key and cfg.get("tools", {}).get("search_api"):
        cfg["tools"]["search_api"]["api_key"] = tool_key

    agent = _build_langchain_agent(cfg)


    prompt_template = cfg["prompt"]["template"]
    

    for paper in papers:
        context = prompt_template.format(overview=overview_text, title=paper.title, abstract=paper.summary)
        resp = agent.invoke({"messages": [{"role": "user", "content": context}]})
        structured_response = _struct_resp2dict(resp.get("structured_response"))
        logging.info(f"LLM rerank response for {paper.arxiv_id}: {structured_response}")

        if structured_response:
            normalized = normalize_llm_rerank_output(structured_response)
            apply_llm_rerank_result(paper, normalized)
        else:
            logging.warning("LLM rerank failed for %s: no structured response", paper.arxiv_id)
            mark_llm_rerank_failed(paper)
    return papers
