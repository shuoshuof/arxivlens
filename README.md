# Overview -> Embedding -> LLM Rerank Demo (CLI)

This repo root contains a minimal demo pipeline extracted from `zotero-arxiv-daily`:

1) Read `overview.md` as project context
2) Fetch arXiv candidates (title + abstract)
3) Embedding rerank via `utils.recommender.rerank_paper`
4) Optional LLM rerank via Ollama (JSON output)
5) Print Top-N to terminal (no email, no web)

## Quick Run

```bash
export ARXIV_QUERY="cs.RO+cs.AI"
python main.py \
  --overview_path overview.md \
  --top_retrieve 50 \
  --enable_llm_rerank true \
  --llm_rerank_backend ollama \
  --ollama_model "qwen2.5:7b"
```

Langflow LLM rerank option:
1) Open Langflow UI and import `llm_rerank_flow.json`
2) Set the Language Model provider to Ollama (or your choice)
3) Copy the flow ID and set `LLM_RERANK_FLOW_ID`
4) Run:
```bash
export LLM_RERANK_FLOW_ID="your-flow-id"
python main.py --enable_llm_rerank true --llm_rerank_backend langflow --llm_rerank_flow_id "$LLM_RERANK_FLOW_ID"
```

## CLI Arguments

- `--overview_path` (default `overview.md`)
- `--arxiv_query` (default from `ARXIV_QUERY`)
- `--top_retrieve` (default `50`)
- `--enable_llm_rerank` (default `true`)
- `--llm_rerank_backend` (`ollama` or `langflow`, default `ollama`)
- `--ollama_base_url` (default `http://localhost:11434`)
- `--ollama_model` (default `qwen2.5:14b`)
- `--langflow_base_url` (default `http://localhost:7863`)
- `--llm_rerank_flow_id` (required for langflow rerank)
- `--langflow_api_key` (optional)
- `--seed` (optional)
- `--debug`

## API Notes

### main.py
- Orchestrates the demo flow.
- Scoring:
  - Embed scores normalized to [0,1] over `top_retrieve`.
  - If LLM rerank enabled: `final = 0.6 * norm_embed + 0.4 * (fit_score / 10)`.
  - If LLM rerank disabled: `final = norm_embed`.
  - When LLM rerank enabled, only `relevant=true` papers are kept.

### utils/arxiv_fetcher.py
- `get_arxiv_paper(query, days=7, max_candidates=150, debug=False) -> list[ArxivPaper]`
- Uses RSS to collect IDs, then fetches metadata in batches of 20.
- Filters by `days` using published time (UTC).

### utils/recommender.py
- `rerank_paper(candidate, corpus, model="avsolatorio/GIST-small-Embedding-v0")`
- `corpus` must be:
  ```python
  [{"data": {"abstractNote": "...", "dateAdded": "YYYY-MM-DDTHH:MM:SSZ"}}]
  ```
- Produces `paper.score` and returns sorted list.

### utils/paper.py
- `ArxivPaper` wrapper fields:
  - `title`, `summary`, `authors`, `arxiv_id`, `url`, `categories`
  - `published`, `published_date`, `pdf_url`
  - scoring/LLM rerank fields: `score`, `final_score`, `llm_rerank_relevant`, `llm_rerank_fit_score`, `llm_rerank_reasons`, `llm_rerank_action`, `llm_rerank_failed`

### backend/ollama_client.py
- `chat_json(model, system, user, base_url="http://localhost:11434", timeout=90, retries=2) -> dict`
- Enforces JSON-only output; retries if invalid JSON.

### backend/langflow_client.py
- `llm_rerank_json(flow_id, overview, title, abstract, base_url="http://localhost:7863", api_key=None, timeout=90, retries=1) -> dict`
- Runs a Langflow flow and extracts JSON output.

### backend/llm_rerank.py
- `ollama_llm_rerank(overview_text, papers, model, base_url, timeout=90, retries=1)`
- `langflow_llm_rerank(overview_text, papers, flow_id, base_url, api_key=None, timeout=90, retries=1)`
- Expects JSON:
  ```json
  {"relevant": true, "fit_score": 0-10, "reasons": ["..."], "action": "..."}
  ```
- Normalizes values and attaches to each paper.
