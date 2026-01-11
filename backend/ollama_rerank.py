from __future__ import annotations

import logging

from backend.ollama_client import chat_json
from backend.rerank_utils import (
    apply_llm_rerank_result,
    build_messages,
    mark_llm_rerank_failed,
    normalize_llm_rerank_output,
)
from utils.paper import ArxivPaper


def ollama_llm_rerank(
    overview_text: str,
    papers: list[ArxivPaper],
    model: str,
    base_url: str,
    timeout: int = 90,
    retries: int = 1,
) -> list[ArxivPaper]:
    for paper in papers:
        system, user = build_messages(overview_text, paper)
        try:
            data = chat_json(
                model=model,
                system=system,
                user=user,
                base_url=base_url,
                timeout=timeout,
                retries=retries,
            )
            normalized = normalize_llm_rerank_output(data)
            apply_llm_rerank_result(paper, normalized)
        except Exception as exc:
            logging.warning("LLM rerank failed for %s: %s", paper.arxiv_id, exc)
            mark_llm_rerank_failed(paper)
    return papers
