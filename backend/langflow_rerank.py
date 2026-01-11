from __future__ import annotations

import logging

from backend.langflow_client import llm_rerank_json
from backend.rerank_utils import (
    apply_llm_rerank_result,
    mark_llm_rerank_failed,
    normalize_llm_rerank_output,
)
from utils.paper import ArxivPaper


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
            normalized = normalize_llm_rerank_output(data)
            apply_llm_rerank_result(paper, normalized)
        except Exception as exc:
            logging.warning("LLM rerank failed for %s: %s", paper.arxiv_id, exc)
            mark_llm_rerank_failed(paper)
    return papers
