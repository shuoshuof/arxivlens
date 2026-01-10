import arxiv
import datetime
import logging
import feedparser
from tqdm import tqdm

from utils.paper import ArxivPaper


def _entry_datetime(entry) -> datetime.datetime | None:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not parsed:
        return None
    return datetime.datetime(*parsed[:6], tzinfo=datetime.timezone.utc)


def _latest_two_days_entry_ids(feed) -> list[str]:
    entries_with_dt: list[tuple[datetime.datetime, str]] = []
    for entry in feed.entries:
        published_at = _entry_datetime(entry)
        if published_at:
            entries_with_dt.append(
                (published_at, entry.id.removeprefix("oai:arXiv.org:"))
            )
    if not entries_with_dt:
        return []
    latest_date = max(dt for dt, _ in entries_with_dt).date()
    second_date = latest_date - datetime.timedelta(days=1)
    return [
        entry_id
        for dt, entry_id in entries_with_dt
        if dt.date() in (latest_date, second_date)
    ]


def _normalize_query_for_api(query: str) -> str:
    cleaned = query.strip()
    if not cleaned:
        return cleaned
    parts = [part for part in cleaned.split("+") if part]
    if not parts:
        return cleaned
    if all("." in part and ":" not in part for part in parts):
        return " OR ".join(f"cat:{part}" for part in parts)
    if "+" in cleaned and " " not in cleaned:
        return cleaned.replace("+", " ")
    return cleaned


def _search_latest_two_days_papers(
    client: arxiv.Client,
    query: str,
    max_results: int = 200,
) -> list[ArxivPaper]:
    search_query = _normalize_query_for_api(query)
    search = arxiv.Search(
        query=search_query,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
        max_results=max_results,
    )
    recent: list[ArxivPaper] = []
    latest_date: datetime.date | None = None
    allowed_dates: set[datetime.date] = set()
    for result in client.results(search):
        paper = ArxivPaper(result)
        published = paper.published
        if published is None:
            continue
        paper_date = published.date()
        if latest_date is None:
            latest_date = paper_date
            allowed_dates = {latest_date, latest_date - datetime.timedelta(days=1)}
        if paper_date in allowed_dates:
            recent.append(paper)
        elif latest_date is not None and paper_date < min(allowed_dates):
            break
    return recent


def get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]:
    client = arxiv.Client(num_retries=10, delay_seconds=1)
    feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
    if "Feed error for query" in feed.feed.title:
        raise ValueError(f"Invalid ARXIV_QUERY: {query}.")

    if debug:
        search = arxiv.Search(query="cat:cs.AI", sort_by=arxiv.SortCriterion.SubmittedDate)
        papers: list[ArxivPaper] = []
        for result in client.results(search):
            papers.append(ArxivPaper(result))
            if len(papers) == 5:
                break
        return papers

    papers: list[ArxivPaper] = []
    all_paper_ids = [
        entry.id.removeprefix("oai:arXiv.org:")
        for entry in feed.entries
        if entry.arxiv_announce_type == "new"
    ]
    if not all_paper_ids:
        fallback_ids = _latest_two_days_entry_ids(feed)
        if fallback_ids:
            logging.info(
                "No new arXiv papers found in RSS; falling back to latest two days (%s items).",
                len(fallback_ids),
            )
            all_paper_ids = fallback_ids
        else:
            logging.info(
                "No new arXiv papers found in RSS; falling back to API search for latest two days."
            )
            fallback_papers = _search_latest_two_days_papers(client, query)
            if fallback_papers:
                return fallback_papers
    bar = tqdm(total=len(all_paper_ids), desc="Retrieving arXiv papers")
    for i in range(0, len(all_paper_ids), 20):
        search = arxiv.Search(id_list=all_paper_ids[i : i + 20])
        batch = [ArxivPaper(p) for p in client.results(search)]
        papers.extend(batch)
        bar.update(len(batch))
    bar.close()
    return papers
