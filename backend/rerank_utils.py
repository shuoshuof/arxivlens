from __future__ import annotations

from typing import Any

from utils.paper import ArxivPaper


def build_messages(overview_text: str, paper: ArxivPaper) -> tuple[str, str]:
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


def normalize_llm_rerank_output(data: dict[str, Any]) -> dict[str, Any]:
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


def apply_llm_rerank_result(paper: ArxivPaper, normalized: dict[str, Any]) -> None:
    paper.llm_rerank_relevant = normalized["relevant"]
    paper.llm_rerank_fit_score = normalized["fit_score"]
    paper.llm_rerank_reasons = normalized["reasons"]
    paper.llm_rerank_action = normalized["action"]


def mark_llm_rerank_failed(paper: ArxivPaper) -> None:
    paper.llm_rerank_failed = True
    paper.llm_rerank_relevant = False
    paper.llm_rerank_fit_score = 0.0
    paper.llm_rerank_reasons = []
    paper.llm_rerank_action = ""
