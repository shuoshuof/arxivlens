# API Technical Documentation

This document describes the root-level demo pipeline: overview -> embedding rerank -> optional LLM rerank -> CLI output. It covers every public function, data shape, and scoring formula.

## Data Model

### `utils.paper.ArxivPaper`
Wrapper around `arxiv.Result` with extra scoring fields.

Properties:
- `title: str` - Paper title.
- `summary: str` - Abstract text.
- `authors: list[str]` - Author names as strings.
- `arxiv_id: str` - ID without version suffix (e.g., `2401.12345`).
- `url: str` - Entry URL (abs page).
- `categories: list[str]` - arXiv categories.
- `published: datetime | None` - Published timestamp (UTC normalized).
- `published_date: str | None` - `YYYY-MM-DD` (UTC).
- `pdf_url: str | None` - PDF URL (derived if missing).

Scoring fields (mutated by pipeline):
- `score: float | None` - Embedding relevance score (from `utils.recommender.rerank_paper`).
- `final_score: float | None` - Final score after normalization and optional LLM rerank fusion.
- `llm_rerank_relevant: bool | None` - LLM rerank relevance flag.
- `llm_rerank_fit_score: float | None` - LLM rerank fit score (0-10).
- `llm_rerank_reasons: list[str] | None` - LLM rerank reasons.
- `llm_rerank_action: str | None` - LLM rerank action suggestion.
- `llm_rerank_failed: bool` - True if LLM rerank call failed.

## Pipeline Entry Point

### `main.py`
CLI entrypoint that orchestrates the full flow.

CLI arguments (also read from env by uppercase name):
- `--overview_path` (default `overview.md`)
- `--arxiv_query` (required if `ARXIV_QUERY` not set)
- `--top_retrieve` (default `50`)
- `--enable_llm_rerank` (default `true`)
- `--llm_rerank_backend` (`ollama`, `langflow`, or `langchain`, default `ollama`)
- `--ollama_base_url` (default `http://localhost:11434`)
- `--ollama_model` (default `qwen2.5:14b`)
- `--langflow_base_url` (default `http://localhost:7863`)
- `--langflow_mode` (`http` or `local`, default `local`)
- `--langflow_flow_id` (required for `langflow` backend)
- `--langflow_flow_path` (flow JSON path for `langflow_mode=local`, default `data/llm_rerank_flow.json`)
- `--langflow_api_key` (optional)
- `--seed` (optional)
- `--debug` (flag)

Notes:
- Langflow rerank expects the `arxiv` conda environment.
- LangChain rerank expects the `lc` conda environment.
- LangChain rerank uses SearchApi; set `SEARCHAPI_API_KEY` (or `SEARCHAPI_KEY`) for tool calls.

Functions:

#### `_str2bool(value: str | bool) -> bool`
Parses boolean CLI/env values. Accepts `true/1/yes/y` (case-insensitive).

#### `add_argument(*args, **kwargs)`
Adds an argparse entry and applies environment-variable defaults:
- For `--foo_bar`, it reads `FOO_BAR` from the environment.
- If the env var is set, it is cast to the arg `type` and used as the default.

#### `load_overview_as_corpus(overview_path: str) -> tuple[list[dict], str]`
Reads `overview.md` and returns:
- `corpus`: a single-item list in the recommender format:
  ```python
  [{"data": {"abstractNote": overview_text, "dateAdded": "YYYY-MM-DDTHH:MM:SSZ"}}]
  ```
- `overview_text`: raw text used for LLM rerank context.

#### `normalize_scores(scores: list[float]) -> list[float]`
Min-max normalization:
- If all scores are equal, returns a list of `1.0`.
- Else:
  ```
  norm_i = (score_i - min_score) / (max_score - min_score)
  ```

#### `format_paper_line(paper: ArxivPaper, rank: int) -> str`
Returns a multi-line string for CLI output. Includes:
- final/embed/llm scores
- published date + categories
- title, URL, PDF URL
- up to 2 LLM rerank reasons + action (if present)

Scoring and filtering (main flow):
1) `ranked = rerank_paper(candidates, corpus)`
2) `top_retrieve = ranked[:top_retrieve]`
3) Normalize embedding scores:
   ```
   norm_embed_i = normalize(score_i)
   final_i = norm_embed_i
   ```
4) If LLM rerank enabled:
   ```
   final_i = 0.6 * norm_embed_i + 0.4 * (fit_score_i / 10)
   keep only llm_rerank_relevant == True
   ```
5) Sort by `final_score` desc, print all.

## arXiv Fetch

### `utils.arxiv_fetcher.get_arxiv_paper(query: str, debug: bool = False) -> list[ArxivPaper]`
Fetches candidate papers using arXiv RSS + API.

Behavior:
- RSS: `https://rss.arxiv.org/atom/{query}`
- Only keeps entries with `arxiv_announce_type == "new"`.
- Batch fetches metadata by ID (20 per request).
- Uses `arxiv.Client(num_retries=10, delay_seconds=1)`.
- Shows progress with `tqdm`.
- `debug=True`: fetches 5 results from `cat:cs.AI` regardless of RSS.

Exceptions:
- Raises `ValueError` if the RSS feed reports a query error.

## Embedding Rerank

### `utils.recommender.rerank_paper(candidate, corpus, model="avsolatorio/GIST-small-Embedding-v0")`
Ranks candidates by similarity to the corpus (overview).

Inputs:
- `candidate: list[ArxivPaper]`
- `corpus: list[dict]` where each item is:
  ```python
  {"data": {"abstractNote": "<text>", "dateAdded": "YYYY-MM-DDTHH:MM:SSZ"}}
  ```
- `model`: SentenceTransformer model name.

Steps:
1) Encode corpus abstracts and candidate summaries.
2) Compute similarity matrix:
   ```
   sim = encoder.similarity(candidate_embeddings, corpus_embeddings)
   ```
3) Time-decay weighting over corpus items:
   ```
   w_j = 1 / (1 + log10(j + 1))
   w = w / sum(w)
   ```
   Corpus is sorted newest -> oldest by `dateAdded` before weighting.
4) Candidate score:
   ```
   score_i = 10 * sum_j(sim[i, j] * w_j)
   ```
5) Writes `paper.score` and returns candidates sorted by score desc.

Notes:
- With a single-item corpus (overview), the weight is always 1.
- `encoder.similarity` uses cosine similarity in SentenceTransformers.

## LLM Rerank

### `backend.llm_rerank.ollama_llm_rerank(overview_text, papers, model, base_url, timeout=90, retries=1)`
Runs Ollama checks on top candidates and attaches structured results.

### `backend.llm_rerank.langflow_llm_rerank(overview_text, papers, flow_id, base_url, api_key=None, timeout=90, retries=1, mode="local", flow_path=None)`
Runs a Langflow flow for each top candidate and attaches structured results.

For each paper:
1) Calls the selected backend function.
2) Normalizes output and sets:
   - `llm_rerank_relevant`
   - `llm_rerank_fit_score`
   - `llm_rerank_reasons`
   - `llm_rerank_action`
3) On failure:
   - `llm_rerank_failed = True`
   - `llm_rerank_relevant = False`
   - `llm_rerank_fit_score = 0.0`
   - `llm_rerank_reasons = []`
   - `llm_rerank_action = ""`

Langflow requirements:
- Import `llm_rerank_flow.json` into Langflow.
- The flow must return JSON with keys:
  ```
  relevant (bool), fit_score (0-10), reasons (list[str]), action (str)
  ```

#### `_normalize_llm_rerank_output(data: dict) -> dict`
Normalization rules:
- `relevant`: accepts bool or string (`"true"`, `"1"`, `"yes"`).
- `fit_score`: cast to float, clamped to `[0, 10]`.
- `reasons`: string -> list; non-list -> empty list; trim empty strings.
- `action`: string, trimmed.

## Ollama Client

### `backend.ollama_client.chat_json(model, system, user, base_url="http://localhost:11434", timeout=90, retries=2) -> dict`
Calls Ollama `/api/chat` and enforces JSON output.

Request payload:
```json
{
  "model": "<model>",
  "messages": [{"role":"system","content":"..."},{"role":"user","content":"..."}],
  "stream": false,
  "format": "json"
}
```

Response parsing:
- Reads `response.json()["message"]["content"]`.
- Parses JSON; if invalid, tries substring from first `{` to last `}`.
- On failure, retries by appending a strict "JSON only" message.
- Raises `RuntimeError` after retries are exhausted.

## Langflow Client

### `backend.langflow_client.langflow_rerank_json_local(overview, title, abstract, flow_path, retries=1) -> dict`

### `backend.langflow_client.langflow_rerank_json_http(flow_id, overview, title, abstract, base_url="http://localhost:7863", api_key=None, timeout=90, retries=1) -> dict`

### `backend.langflow_client.llm_rerank_json(flow_id, overview, title, abstract, base_url="http://localhost:7863", api_key=None, timeout=90, retries=1, mode="local", flow_path=None) -> dict`
Calls Langflow `/api/v1/run/{flow_id}` and extracts JSON output.

## Scoring Summary

Embedding relevance:
```
score_i = 10 * sum_j(sim[i, j] * w_j)
w_j = 1 / (1 + log10(j + 1))
```

Normalization:
```
norm_i = (score_i - min(score)) / (max(score) - min(score))
if all equal -> norm_i = 1.0
```

Final score:
- LLM rerank disabled:
  ```
  final_i = norm_i
  ```
- LLM rerank enabled:
  ```
  final_i = 0.6 * norm_i + 0.4 * (fit_score_i / 10)
  keep only llm_rerank_relevant == True
  ```
