from __future__ import annotations

import logging
from typing import Any

from backend.langflow_client import llm_rerank_json
from backend.ollama_client import chat_json
from utils.paper import ArxivPaper


def _build_messages(overview_text: str, paper: ArxivPaper) -> tuple[str, str]:
    system = (
        "You are a research assistant. Decide whether a candidate paper is relevant to the project "
        "overview. Return ONLY JSON with keys: relevant (bool), fit_score (0-10 number), "
        "reasons (list of strings), action (string)."
    )
    user = (
        "Project overview:\n"
        f"{overview_text}\n\n"
        "Candidate paper:\n"
        f"Title: {paper.title}\n"
        f"Abstract: {paper.summary}\n\n"
        "Return JSON only."
    )
    return system, user


def _normalize_llm_rerank_output(data: dict[str, Any]) -> dict[str, Any]:
    relevant = data.get("relevant", False)
    if isinstance(relevant, str):
        relevant = relevant.strip().lower() in {"true", "1", "yes"}
    fit_score = data.get("fit_score", 0)
    try:
        fit_score = float(fit_score)
    except (TypeError, ValueError):
        fit_score = 0.0
    fit_score = max(0.0, min(10.0, fit_score))

    reasons = data.get("reasons", [])
    if isinstance(reasons, str):
        reasons = [reasons]
    if not isinstance(reasons, list):
        reasons = []
    reasons = [str(r).strip() for r in reasons if str(r).strip()]

    action = data.get("action", "")
    action = str(action).strip()

    return {
        "relevant": bool(relevant),
        "fit_score": fit_score,
        "reasons": reasons,
        "action": action,
    }



def ollama_llm_rerank(
    overview_text: str,
    papers: list[ArxivPaper],
    model: str,
    base_url: str,
    timeout: int = 90,
    retries: int = 1,
) -> list[ArxivPaper]:
    for paper in papers:
        system, user = _build_messages(overview_text, paper)
        try:
            data = chat_json(
                model=model,
                system=system,
                user=user,
                base_url=base_url,
                timeout=timeout,
                retries=retries,
            )
            normalized = _normalize_llm_rerank_output(data)
            paper.llm_rerank_relevant = normalized["relevant"]
            paper.llm_rerank_fit_score = normalized["fit_score"]
            paper.llm_rerank_reasons = normalized["reasons"]
            paper.llm_rerank_action = normalized["action"]
        except Exception as exc:
            logging.warning("LLM rerank failed for %s: %s", paper.arxiv_id, exc)
            paper.llm_rerank_failed = True
            paper.llm_rerank_relevant = False
            paper.llm_rerank_fit_score = 0.0
            paper.llm_rerank_reasons = []
            paper.llm_rerank_action = ""
    return papers


def langflow_llm_rerank(
    overview_text: str,
    papers: list[ArxivPaper],
    flow_id: str,
    base_url: str,
    api_key: str | None = None,
    timeout: int = 90,
    retries: int = 1,
    mode: str = "local",
    flow_path: str | None = None,
) -> list[ArxivPaper]:
    for paper in papers:
        try:
            data = llm_rerank_json(
                flow_id=flow_id,
                overview=overview_text,
                title=paper.title,
                abstract=paper.summary,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                retries=retries,
                mode=mode,
                flow_path=flow_path,
            )
            normalized = _normalize_llm_rerank_output(data)
            paper.llm_rerank_relevant = normalized["relevant"]
            paper.llm_rerank_fit_score = normalized["fit_score"]
            paper.llm_rerank_reasons = normalized["reasons"]
            paper.llm_rerank_action = normalized["action"]
        except Exception as exc:
            logging.warning("LLM rerank failed for %s: %s", paper.arxiv_id, exc)
            paper.llm_rerank_failed = True
            paper.llm_rerank_relevant = False
            paper.llm_rerank_fit_score = 0.0
            paper.llm_rerank_reasons = []
            paper.llm_rerank_action = ""
    return papers
