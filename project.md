
# 开发文档：Overview→Embedding初筛→CAG精选（CLI 课程 Demo）

## 0. 背景与目标

我们要在 `zotero-arxiv-daily` 基础上，做一个**极简课程 demo**：

* 输入：`overview.md`（已存在，无需再从 Zotero/网页生成）
* 每次运行：

  1. 从 arXiv 拉取候选论文（title+abstract）
  2. 用 embedding 相似度对候选论文**初筛排序**（例如 Top-50）
  3. 对 Top-50 做 **CAG 精选**：把 `overview.md` 作为上下文，让 LLM 输出是否相关、理由、以及落地建议
  4. 最终**仅在终端打印** Top-N（不生成网页、不发邮件）

说明：上游项目的 `main.py` 是全流程 orchestrator（Zotero→arXiv→排序→邮件）([DeepWiki][1])；我们只保留 arXiv 拉取与排序思路，移除/跳过 Zotero 与邮件流程。

---

## 1. 上游项目关键点（便于最小改动对齐）

### 1.1 主流程能力

上游 `main.py` 的职责包括：环境变量配置、取 Zotero corpus、取 arXiv 新论文、相关性排序、构造并发送邮件([DeepWiki][1])。

其中 arXiv 拉取逻辑：解析 arXiv RSS/Feed、拿新发布论文 ID、批量拉取并构造 `ArxivPaper` 列表([DeepWiki][1])。

### 1.2 排序器接口（我们要复用）

上游推荐引擎是一个函数（DeepWiki 文档里明确）：

* `rerank_paper(candidate: list[ArxivPaper], corpus: list[dict], model: str=...) -> list[ArxivPaper]`([DeepWiki][2])
* 依赖字段：

  * `candidate[i].summary`（论文摘要）
  * `corpus[j]['data']['abstractNote']`（语料摘要）
  * `corpus[j]['data']['dateAdded']`（用于时间衰减权重）([DeepWiki][2])

排序算法：SentenceTransformer 对摘要做 embedding，然后计算相似度矩阵并叠加时间衰减权重（`1/(1+log10(index+1))`），得到每个 candidate 的分数并排序([DeepWiki][2])。

### 1.3 LLM 封装（可扩展用来做 CAG）

上游有一个统一 `LLM` 类，提供 `generate(messages)->text` 的接口，支持“有 API key 用云端、否则用本地 Llama backend”的切换([DeepWiki][3])。我们可以：

* 方案 A：扩展它新增 **Ollama backend**
* 方案 B：新写一个 `ollama_client.py` 专门负责 CAG（更快更稳）

### 1.4 重要环境变量（可沿用/新增）

上游 README 指明 `ARXIV_QUERY` 是必填，且用 `+` 拼接多个类别（示例：`cs.AI+cs.CV+cs.LG+cs.CL`）([GitHub][4])；`MAX_PAPER_NUM` 会影响生成 TL;DR 的运行时间等([GitHub][4])。我们 demo 不发邮件、不做 TL;DR，但仍可沿用 `ARXIV_QUERY` 的配置方式。

---

## 2. 新增“Overview Demo 模式”的设计

### 2.1 功能需求（必须实现）

1. **读取 overview.md** 作为项目上下文（不再访问 Zotero）
2. **抓取 arXiv 候选论文**（title+abstract 即可）
3. **Embedding 初筛**：用上游 `rerank_paper`，但 corpus 改为“仅一条 overview”
4. **CAG 精选**：对 Top-50 逐篇调用 LLM（Ollama），输出结构化判断
5. **终端输出**：打印最终 Top-N（含分数、标题、日期、链接、理由/建议）

### 2.2 非需求（明确不做）

* 不生成网页/前端
* 不发送邮件（跳过 SMTP、construct_email 等）
* 不解析 PDF/LaTeX（上游 paper processing 很丰富，但 demo 可不启用）([DeepWiki][5])

---

## 3. 关键实现方案（最小改动）

### 3.1 “伪 corpus” 注入（复用 rerank_paper，不改推荐器）

构造一个只有 1 条的 corpus：

```python
corpus = [{
  "data": {
    "abstractNote": overview_text,
    "dateAdded": "<now in ISO8601>"
  }
}]
```

理由：上游推荐器正是从 `paper['data']['abstractNote']` 取文本 embedding，且 corpus 字段要求很明确([DeepWiki][2])。这样我们几乎不用动 recommender，只把 Zotero corpus 换成 overview corpus。

> 注意：推荐器内部有“按 dateAdded 新旧排序 + 时间衰减权重”([DeepWiki][2])。只有 1 条 corpus 时，衰减权重恒为 1，不影响逻辑。

### 3.2 CAG 精选：Ollama Chat 输出 JSON

对 embedding Top-50 的每篇论文，调用一次 Ollama（/api/chat），prompt 固定注入 overview：

**输入：**

* Context：`overview.md`
* Candidate：论文 title + abstract

**输出（必须是 JSON，便于程序二次排序和过滤）：**

```json
{
  "relevant": true,
  "fit_score": 0-10,
  "reasons": ["...", "..."],
  "action": "一句话：如何用到项目/建议读到哪"
}
```

最终分数建议：

* `final = 0.6 * normalize(embed_score) + 0.4 * (fit_score/10)`
* 只保留 `relevant=true`

（权重可调；demo 推荐这个平衡：既体现 embedding，又体现 CAG 判断。）

---

## 4. 代码改动清单（面向 code agent 的任务列表）

### 4.1 新增/修改 CLI 参数

在 `main.py`（或新建 `demo_main.py`）增加参数：

* `--overview_path`（默认 `overview.md`）
* `--arxiv_query`（可选，默认读取环境变量 `ARXIV_QUERY`）
* `--days`（默认 7；用于限制候选论文发布时间窗口）
* `--max_candidates`（默认 150；先拉这么多候选再过滤 days）
* `--top_retrieve`（默认 50；embedding 初筛保留数量）
* `--top_print`（默认 10；终端输出数量）
* `--enable_cag`（默认 true）
* `--ollama_base_url`（默认 `http://localhost:11434`）
* `--ollama_model`（默认你本地可用的 chat 模型名）
* `--seed`（可选，确保 demo 稳定）

### 4.2 新增模块：`cag_refine.py`

职责：输入 `overview_text` 与 `papers_top_retrieve`，输出 `papers_refined`（附加字段：`cag_relevant, cag_fit_score, cag_reasons, cag_action, final_score`）。

要求：

* 支持并发（可选）：但为了 demo 稳定可以先串行
* 失败重试（建议 1-2 次）
* JSON 解析失败要兜底（重新让模型按 JSON 输出；或者标记该 paper cag_failed）

### 4.3 新增模块：`ollama_client.py`

提供：

* `chat_json(model, system, user) -> dict`
* 做好：

  * HTTP 请求封装
  * 超时
  * 失败处理
  * 强制 JSON 输出策略（比如 system 里强调 “只输出 JSON”）

> 如果你更想复用上游 `LLM.generate()` 统一接口，也可以扩展 `llm.py` 新增 Ollama backend；上游已有 “统一 generate 接口” 的设计思路([DeepWiki][3])。

### 4.4 改造 `main.py`（或新建 demo 入口）

把原本 `get_zotero_corpus` 的调用替换为 `load_overview_as_corpus()`。

上游 `main.py` 流程是线性的 orchestrator（取 Zotero→取 arXiv→排序→邮件）([DeepWiki][1])；我们做一个分支模式：

* `if DEMO_MODE:`

  * `corpus = [overview_as_one_item]`
  * `candidates = get_arxiv_paper(...)`（可复用上游 arXiv 拉取策略([DeepWiki][1])）
  * `ranked = rerank_paper(candidates, corpus, model=...)`（复用推荐器接口([DeepWiki][2])）
  * `top_retrieve = ranked[:top_retrieve]`
  * `refined = cag_refine(overview_text, top_retrieve)`（可开关）
  * `print_cli(refined[:top_print])`
  * **return**（不走 email 相关逻辑）

### 4.5 输出格式（终端打印）

每篇输出建议包括：

* `rank`
* `final_score`、`embed_score`、`fit_score`
* `published`、`categories`
* `title`
* `url` / `pdf_url`
* `reasons`（两条）
* `action`（一句）

---

## 5. 验收标准（必须满足）

1. 在没有 Zotero 凭证的情况下运行成功（不依赖 `ZOTERO_ID/ZOTERO_KEY`）
2. 能正确读取 `overview.md`
3. 能按 `ARXIV_QUERY` 拉取论文（类别拼接规则沿用上游：用 `+` 拼接）([GitHub][4])
4. embedding 初筛后输出 Top-50（至少能看到排序分数）
5. CAG 精选后，终端输出 Top-10，且每篇有 `relevant/fit_score/reasons/action`
6. 全流程耗时可控（demo 建议：`max_candidates<=150`，`top_retrieve<=50`）

---

## 6. 运行说明（给 TA/老师演示用）

### 6.1 本地依赖（建议）

* Python + 项目依赖（上游 README 也支持本地运行方式）([GitHub][4])
* 启动 Ollama：`ollama serve`
* 拉一个 chat 模型（你本地已有即可）

### 6.2 示例命令

```bash
export ARXIV_QUERY="cs.RO+cs.AI"
python main.py \
  --overview_path overview.md \
  --days 7 \
  --max_candidates 150 \
  --top_retrieve 50 \
  --top_print 10 \
  --enable_cag true \
  --ollama_model "qwen2.5:7b"
```

---

## 7. 可选加分项（不影响主线）

* CAG 只对 Top-20 做（更快），其余仅 embedding 输出
* 对 `fit_score` 做一致性：重复调用一次取平均（更稳但更慢）
* 加一个 `--with_keywords`：从 overview 提取关键词，用于解释 reasons（不靠 LLM 也能解释）

---

## 8. 实施优先级（建议按这个顺序开发）

1. DEMO_MODE 路径能跑通：overview→arXiv→rerank→打印（先不做 CAG）
2. 加入 Ollama CAG：Top-50 逐篇判断，输出 JSON
3. 加最终排序融合（final_score）+ 输出美化

