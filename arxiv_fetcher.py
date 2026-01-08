import arxiv
import feedparser
from tqdm import tqdm

from paper import ArxivPaper


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
    bar = tqdm(total=len(all_paper_ids), desc="Retrieving arXiv papers")
    for i in range(0, len(all_paper_ids), 20):
        search = arxiv.Search(id_list=all_paper_ids[i : i + 20])
        batch = [ArxivPaper(p) for p in client.results(search)]
        papers.extend(batch)
        bar.update(len(batch))
    bar.close()
    return papers
