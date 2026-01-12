for i in {1..10}; do
  echo "==== Run $i/10 ===="
  python main.py --overview_path data/overview.md --arxiv_query "cs.AI+cs.CV+cs.LG+cs.CL" --top_retrieve 10 --enable_llm_rerank true --llm_rerank_backend langchain 2>&1 | grep --line-buffered -E "WARNING" || echo "Run $i failed (exit $?)"
done