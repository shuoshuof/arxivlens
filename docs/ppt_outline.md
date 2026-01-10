# PPT Outline (ArxivLens)

## Slide 1 - Title
- Team name: SignalForge
- Project title: ArxivLens - Overview to Embedding to LLM Rerank
- Tagline (open strong): "Turn the arXiv firehose into a ranked short list in minutes."
- Team photos (professional headshots, name under each):
  - ![<<Name1>>](path/to/photo1.jpg)
    <<Name1>>
  - ![<<Name2>>](path/to/photo2.jpg)
    <<Name2>>
  - ![<<Name3>>](path/to/photo3.jpg)
    <<Name3>>

Notes (not on slide):
- <<Name1>> opens with the hook and 1 sentence on why this matters.

## Slide 2 - Problem and Why It Matters
- arXiv publishes a high volume of new papers every day.
- In fast moving fields like RL, robotics, and embodied AI, missing a key paper can derail a project.
- Manual screening does not scale and is inconsistent across reviewers.

Notes (not on slide):
- <<Name2>>: 1 to 2 sentences on the cost of missed or late discoveries.

## Slide 3 - Current Process Pain Points (Based on Research)
- Manual RSS browsing and keyword filters miss relevant work.
- Abstract skim decisions vary by person and time pressure.
- Little traceability: no consistent reasons or scores for why a paper was picked.
- Time sink: repeated reading of similar abstracts.

Notes (not on slide):
- <<Name3>>: 1 to 2 sentences on where time is lost today.

## Slide 4 - Redesigned Process and What the LLM Does
- Input: project overview (overview.md) + daily arXiv query.
- Step 1: embedding rerank finds topical similarity.
- Step 2: LLM rerank judges relevance and returns reasons + action in JSON.
- Output: ranked short list with scores and explainable reasons.

Notes (not on slide):
- <<Name1>>: 1 to 2 sentences on why the LLM step adds value beyond embeddings.

## Slide 5 - Expected Benefits
- Faster triage: reduce screening time to minutes.
- Better alignment: stronger match to project goals and domains.
- Explainability: reasons and action make decisions reviewable.
- Consistency: standardized scoring across the team.

Notes (not on slide):
- <<Name2>>: 1 to 2 sentences on team impact and decision quality.

## Slide 6 - Technology (Keep It Short)
- Data plan: overview.md as the project corpus + fresh arXiv candidates.
- Retrieval: SentenceTransformer embedding rerank (GIST-small).
- LLM: local model via Ollama (qwen2.5) or Langflow flow for JSON output.
- Fusion: final score = 0.6 * embed + 0.4 * LLM fit score.

Notes (not on slide):
- <<Name3>>: 1 sentence on why local LLM and JSON output keep it reliable.

## Slide 7 - Risks and Safeguards
- Risk: LLM hallucination or inconsistent outputs.
- Safeguard: strict JSON schema with retries and normalization.
- Risk: false positives or negatives.
- Safeguard: keep embedding score as anchor and require relevance=true.
- Risk: privacy concerns.
- Safeguard: local LLM option with no external calls.

Notes (not on slide):
- <<Name1>>: 1 to 2 sentences on how safeguards keep quality and trust.

## Slide 8 - Evaluation Plan
- Build a labeled set of recent papers (relevant or not).
- Compare baseline (embeddings only) vs LLM rerank.
- Metrics: precision, recall, time to decide, reviewer agreement.
- Pilot: weekly review with the team and iterate prompts.

Notes (not on slide):
- <<Name2>>: 1 to 2 sentences on how we will prove it works.

## Slide 9 - Conclusion
- Why LLM: adds judgment, reasons, and action on top of similarity.
- Result: faster, more reliable literature screening for our project focus.
- Ask: approve pilot run for the next arXiv cycle.

Notes (not on slide):
- <<Name3>>: 1 sentence closing and handoff.
