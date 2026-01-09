from __future__ import annotations

from datetime import datetime, timezone
from functools import cached_property
from typing import Optional
import re

import arxiv


class ArxivPaper:
    def __init__(self, paper: arxiv.Result):
        self._paper = paper
        self.score: Optional[float] = None
        self.llm_rerank_relevant: Optional[bool] = None
        self.llm_rerank_fit_score: Optional[float] = None
        self.llm_rerank_reasons: Optional[list[str]] = None
        self.llm_rerank_action: Optional[str] = None
        self.llm_rerank_failed: bool = False
        self.final_score: Optional[float] = None

    @property
    def title(self) -> str:
        return self._paper.title

    @property
    def summary(self) -> str:
        return self._paper.summary

    @property
    def authors(self) -> list[str]:
        return [str(a) for a in self._paper.authors]

    @cached_property
    def arxiv_id(self) -> str:
        return re.sub(r"v\d+$", "", self._paper.get_short_id())

    @property
    def url(self) -> str:
        return self._paper.entry_id

    @property
    def categories(self) -> list[str]:
        return list(self._paper.categories or [])

    @property
    def published(self) -> Optional[datetime]:
        published = getattr(self._paper, "published", None)
        if published is None:
            return None
        if published.tzinfo is None:
            return published.replace(tzinfo=timezone.utc)
        return published

    @property
    def published_date(self) -> Optional[str]:
        published = self.published
        if published is None:
            return None
        return published.astimezone(timezone.utc).strftime("%Y-%m-%d")

    @property
    def pdf_url(self) -> Optional[str]:
        pdf_url = getattr(self._paper, "pdf_url", None)
        if pdf_url:
            return pdf_url
        if getattr(self._paper, "links", None):
            for link in self._paper.links:
                if "pdf" in link.href:
                    self._paper.pdf_url = link.href
                    return link.href
        pdf_url = f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
        self._paper.pdf_url = pdf_url
        return pdf_url
