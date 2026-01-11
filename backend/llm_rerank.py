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
    "langchain_llm_rerank",
]

_build_messages = build_messages
_normalize_llm_rerank_output = normalize_llm_rerank_output




def langchain_llm_rerank(
    overview_text: str,
    papers: list[ArxivPaper],
    *args: Any,
    **kwargs: Any,
) -> list[ArxivPaper]:
    from backend.langchain_rerank import langchain_llm_rerank as _impl

    return _impl(overview_text, papers, *args, **kwargs)
