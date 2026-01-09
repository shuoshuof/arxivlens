import argparse
import logging
import os
import random
from datetime import datetime, timezone

from dotenv import load_dotenv

from utils.arxiv_fetcher import get_arxiv_paper
from backend.llm_rerank import langflow_llm_rerank, ollama_llm_rerank
from utils.recommender import rerank_paper


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"true", "1", "yes", "y"}


parser = argparse.ArgumentParser(description="Overview → Embedding → LLM rerank demo")


def add_argument(*args, **kwargs):
    def get_env(key: str, default=None):
        value = os.environ.get(key)
        if value == "" or value is None:
            return default
        return value

    parser.add_argument(*args, **kwargs)
    arg_full_name = kwargs.get("dest", args[-1][2:])
    env_name = arg_full_name.upper()
    env_value = get_env(env_name)
    if env_value is not None:
        if kwargs.get("type") == _str2bool:
            env_value = _str2bool(env_value)
        elif kwargs.get("type") is not None:
            env_value = kwargs.get("type")(env_value)
        parser.set_defaults(**{arg_full_name: env_value})


def load_overview_as_corpus(overview_path: str) -> tuple[list[dict], str]:
    with open(overview_path, "r", encoding="utf-8") as file:
        overview_text = file.read().strip()
    if not overview_text:
        raise ValueError(f"Overview file is empty: {overview_path}")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    corpus = [{"data": {"abstractNote": overview_text, "dateAdded": now}}]
    return corpus, overview_text


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score - min_score < 1e-9:
        return [1.0 for _ in scores]
    return [(s - min_score) / (max_score - min_score) for s in scores]


def format_paper_line(paper, rank: int) -> str:
    embed_score = paper.score if paper.score is not None else 0.0
    fit_score = (
        paper.llm_rerank_fit_score
        if paper.llm_rerank_fit_score is not None
        else 0.0
    )
    final_score = paper.final_score if paper.final_score is not None else 0.0
    categories = ", ".join(paper.categories) if paper.categories else "n/a"
    published = paper.published_date or "n/a"
    reasons = paper.llm_rerank_reasons or []
    lines = [
        f"{rank}. final={final_score:.3f} embed={embed_score:.3f} llm={fit_score:.1f}",
        f"   published: {published} | categories: {categories}",
        f"   title: {paper.title}",
        f"   url: {paper.url}",
        f"   pdf: {paper.pdf_url}",
    ]
    for reason in reasons[:2]:
        lines.append(f"   reason: {reason}")
    if paper.llm_rerank_action:
        lines.append(f"   action: {paper.llm_rerank_action}")
    return "\n".join(lines)


if __name__ == "__main__":
    load_dotenv(override=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    add_argument("--overview_path", type=str, help="Path to overview.md", default="overview.md")
    add_argument(
        "--arxiv_query",
        type=str,
        help="Arxiv RSS query, e.g. cs.AI+cs.LG",
        default=None,
    )
    add_argument("--top_retrieve", type=int, help="Top papers after embedding rerank", default=50)
    add_argument(
        "--enable_llm_rerank",
        type=_str2bool,
        help="Enable LLM rerank",
        default=True,
    )
    add_argument(
        "--llm_rerank_backend",
        type=str,
        help="LLM rerank backend: ollama or langflow",
        default="ollama",
    )
    add_argument(
        "--ollama_base_url",
        type=str,
        help="Ollama base URL",
        default="http://localhost:11434",
    )
    add_argument(
        "--ollama_model",
        type=str,
        help="Ollama chat model name",
        default="qwen2.5:14b",
    )
    add_argument(
        "--langflow_base_url",
        type=str,
        help="Langflow base URL",
        default="http://localhost:7863",
    )
    add_argument(
        "--langflow_flow_id",
        type=str,
        help="Langflow flow ID for LLM rerank",
        default=None,
    )
    add_argument(
        "--langflow_api_key",
        type=str,
        help="Langflow API key (if required)",
        default=None,
    )
    add_argument("--seed", type=int, help="Random seed", default=None)
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("arxiv").setLevel(logging.WARNING)

    if args.seed is not None:
        random.seed(args.seed)
        try:
            import numpy as np

            np.random.seed(args.seed)
        except Exception:
            logging.debug("Numpy not available for seeding.")

    if not args.arxiv_query:
        raise ValueError("Missing ARXIV_QUERY. Set --arxiv_query or ARXIV_QUERY env.")

    logging.info("Loading overview from %s", args.overview_path)
    corpus, overview_text = load_overview_as_corpus(args.overview_path)

    logging.info("Retrieving arXiv papers for query: %s", args.arxiv_query)
    candidates = get_arxiv_paper(query=args.arxiv_query, debug=args.debug)
    if not candidates:
        logging.info("No candidates retrieved.")
        raise SystemExit(0)

    logging.info("Reranking %s candidates with embedding model", len(candidates))
    ranked = rerank_paper(candidates, corpus)
    top_retrieve = ranked[: max(0, args.top_retrieve)]
    if not top_retrieve:
        logging.info("No papers left after embedding rerank.")
        raise SystemExit(0)

    embed_scores = [paper.score or 0.0 for paper in top_retrieve]
    normalized_scores = normalize_scores(embed_scores)
    for paper, norm_score in zip(top_retrieve, normalized_scores):
        paper.final_score = norm_score

    if args.enable_llm_rerank:
        backend = (args.llm_rerank_backend or "ollama").strip().lower()
        logging.info("Running LLM rerank via %s", backend)
        if backend == "ollama":
            ollama_llm_rerank(
                overview_text,
                top_retrieve,
                model=args.ollama_model,
                base_url=args.ollama_base_url,
            )
        elif backend == "langflow":
            if not args.langflow_flow_id:
                raise ValueError(
                    "Missing LLM_RERANK_FLOW_ID. Set --langflow_flow_id or LLM_RERANK_FLOW_ID env."
                )
            langflow_llm_rerank(
                overview_text,
                top_retrieve,
                flow_id=args.langflow_flow_id,
                base_url=args.langflow_base_url,
                api_key=args.langflow_api_key,
            )
        else:
            raise ValueError(f"Unsupported LLM rerank backend: {backend}")
        for paper, norm_score in zip(top_retrieve, normalized_scores):
            fit_score = paper.llm_rerank_fit_score or 0.0
            paper.final_score = 0.6 * norm_score + 0.4 * (fit_score / 10.0)
        top_retrieve = [p for p in top_retrieve if p.llm_rerank_relevant]

    top_retrieve = sorted(
        top_retrieve, key=lambda p: p.final_score or 0.0, reverse=True
    )
    if not top_retrieve:
        for idx, paper in enumerate(ranked[: args.top_retrieve], start=1):
            print(format_paper_line(paper, idx))
            print("")
    else:
        logging.info("Printing %s papers", len(top_retrieve))
        for idx, paper in enumerate(top_retrieve, start=1):
            print(format_paper_line(paper, idx))
            print("")
