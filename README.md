# Overview -> Embedding -> CAG Demo (CLI)

This repo root contains a minimal demo pipeline extracted from `zotero-arxiv-daily`:

1) Read `overview.md` as project context
2) Fetch arXiv candidates (title + abstract)
3) Embedding rerank via `recommender.rerank_paper`
4) Optional CAG refinement via Ollama (JSON output)
5) Print Top-N to terminal (no email, no web)

## Quick Run

```bash
export ARXIV_QUERY="cs.RO+cs.AI"
python main.py \
  --overview_path overview.md \
  --top_retrieve 50 \
  --enable_cag true \
  --ollama_model "qwen2.5:7b"
```

## CLI Arguments

- `--overview_path` (default `overview.md`)
- `--arxiv_query` (default from `ARXIV_QUERY`)
- `--top_retrieve` (default `50`)
- `--enable_cag` (default `true`)
- `--ollama_base_url` (default `http://localhost:11434`)
- `--ollama_model` (default `qwen2.5:7b`)
- `--seed` (optional)
- `--debug`

## API Notes

### main.py
- Orchestrates the demo flow.
- Scoring:
  - Embed scores normalized to [0,1] over `top_retrieve`.
  - If CAG enabled: `final = 0.6 * norm_embed + 0.4 * (fit_score / 10)`.
  - If CAG disabled: `final = norm_embed`.
  - When CAG enabled, only `relevant=true` papers are kept.

### arxiv_fetcher.py
- `get_arxiv_paper(query, days=7, max_candidates=150, debug=False) -> list[ArxivPaper]`
- Uses RSS to collect IDs, then fetches metadata in batches of 20.
- Filters by `days` using published time (UTC).

### recommender.py
- `rerank_paper(candidate, corpus, model="avsolatorio/GIST-small-Embedding-v0")`
- `corpus` must be:
  ```python
  [{"data": {"abstractNote": "...", "dateAdded": "YYYY-MM-DDTHH:MM:SSZ"}}]
  ```
- Produces `paper.score` and returns sorted list.

### paper.py
- `ArxivPaper` wrapper fields:
  - `title`, `summary`, `authors`, `arxiv_id`, `url`, `categories`
  - `published`, `published_date`, `pdf_url`
  - scoring/CAG fields: `score`, `final_score`, `cag_relevant`, `cag_fit_score`, `cag_reasons`, `cag_action`, `cag_failed`

### ollama_client.py
- `chat_json(model, system, user, base_url="http://localhost:11434", timeout=90, retries=2) -> dict`
- Enforces JSON-only output; retries if invalid JSON.

### cag_refine.py
- `cag_refine(overview_text, papers, model, base_url, timeout=90, retries=1)`
- Expects JSON:
  ```json
  {"relevant": true, "fit_score": 0-10, "reasons": ["..."], "action": "..."}
  ```
- Normalizes values and attaches to each paper.
