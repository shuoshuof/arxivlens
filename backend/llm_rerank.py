from __future__ import annotations

from typing import Any

from backend.rerank_utils import (  # re-export for backwards compatibility
    apply_llm_rerank_result,
    build_messages,
    mark_llm_rerank_failed,
    normalize_llm_rerank_output,
)
from utils.paper import ArxivPaper

__all__ = [
    "_build_messages",
    "_normalize_llm_rerank_output",
    "apply_llm_rerank_result",
    "build_messages",
    "mark_llm_rerank_failed",
    "normalize_llm_rerank_output",
    "ollama_llm_rerank",
    "langflow_llm_rerank",
    "langchain_llm_rerank",
]

_build_messages = build_messages
_normalize_llm_rerank_output = normalize_llm_rerank_output


def ollama_llm_rerank(
    overview_text: str,
    papers: list[ArxivPaper],
    model: str,
    base_url: str,
    timeout: int = 90,
    retries: int = 1,
) -> list[ArxivPaper]:
    from backend.ollama_rerank import ollama_llm_rerank as _impl

    return _impl(
        overview_text,
        papers,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
    )


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
    from backend.langflow_rerank import langflow_llm_rerank as _impl

    return _impl(
        overview_text,
        papers,
        flow_id=flow_id,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        retries=retries,
        mode=mode,
        flow_path=flow_path,
    )


def langchain_llm_rerank(
    overview_text: str,
    papers: list[ArxivPaper],
    *args: Any,
    **kwargs: Any,
) -> list[ArxivPaper]:
    from backend.langchain_rerank import langchain_llm_rerank as _impl

    return _impl(overview_text, papers, *args, **kwargs)
