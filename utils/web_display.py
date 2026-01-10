from __future__ import annotations

import html
import http.server
import tempfile
from pathlib import Path
from typing import Iterable


def _build_html(papers: Iterable[object]) -> str:
    cards: list[str] = []
    for idx, paper in enumerate(papers, start=1):
        title = html.escape(getattr(paper, "title", "") or "")
        url = html.escape(getattr(paper, "url", "") or "")
        pdf_url = html.escape(getattr(paper, "pdf_url", "") or "")
        paper_link = url or "#"
        pdf_link = pdf_url or url or "#"
        summary = html.escape(getattr(paper, "summary", "") or "")
        reasons = getattr(paper, "llm_rerank_reasons", None) or []
        reasons_items = "".join(
            f"<li>{html.escape(reason)}</li>" for reason in reasons
        )
        if not reasons_items:
            reasons_items = "<li>No recommendation reasons provided.</li>"
        cards.append(
            f"""
            <article class="card" style="--i: {idx};">
              <div class="card-header">
                <span class="rank">#{idx:02d}</span>
                <h2 class="title">{title}</h2>
              </div>
              <div class="links">
                <a href="{paper_link}" target="_blank" rel="noopener">Paper</a>
                <a href="{pdf_link}" target="_blank" rel="noopener">PDF</a>
              </div>
              <p class="summary">{summary}</p>
              <div class="reasons">
                <h3>Recommendation reasons</h3>
                <ul>{reasons_items}</ul>
              </div>
            </article>
            """
        )

    cards_html = "\n".join(cards) if cards else "<p>No papers to display.</p>"
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>ArxivLens Results</title>
    <style>
      :root {{
        --bg: #f6f1e8;
        --ink: #1f1e1c;
        --muted: #5e5a55;
        --accent: #2f6d6a;
        --card: #fffaf1;
        --shadow: rgba(28, 27, 26, 0.12);
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: "Iowan Old Style", "Palatino", "Book Antiqua", "Georgia", serif;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(46, 110, 105, 0.12), transparent 45%),
          radial-gradient(circle at top right, rgba(153, 114, 73, 0.12), transparent 40%),
          var(--bg);
      }}

      header {{
        max-width: 1100px;
        margin: 52px auto 32px;
        padding: 0 24px;
      }}

      h1 {{
        font-size: clamp(2.2rem, 4vw, 3rem);
        margin: 0 0 10px;
        letter-spacing: 0.02em;
      }}

      .subtitle {{
        color: var(--muted);
        margin: 0;
        font-size: 1.05rem;
      }}

      main {{
        max-width: 1100px;
        margin: 0 auto 60px;
        padding: 0 24px 24px;
        display: grid;
        gap: 22px;
      }}

      .card {{
        background: var(--card);
        border-radius: 18px;
        padding: 24px 26px;
        box-shadow: 0 20px 40px var(--shadow);
        border: 1px solid rgba(31, 30, 28, 0.06);
        animation: fadeUp 0.6s ease forwards;
        opacity: 0;
        animation-delay: calc(var(--i) * 0.05s);
      }}

      .card-header {{
        display: flex;
        gap: 14px;
        align-items: baseline;
      }}

      .rank {{
        color: var(--accent);
        font-weight: 700;
        letter-spacing: 0.08em;
        font-size: 0.9rem;
      }}

      .title {{
        margin: 0;
        font-size: 1.35rem;
        line-height: 1.4;
      }}

      .links {{
        margin: 12px 0 16px;
        display: flex;
        gap: 16px;
        font-size: 0.95rem;
      }}

      .links a {{
        color: var(--accent);
        text-decoration: none;
        border-bottom: 1px solid rgba(47, 109, 106, 0.3);
      }}

      .summary {{
        color: var(--muted);
        line-height: 1.6;
        margin: 0 0 18px;
      }}

      .reasons h3 {{
        margin: 0 0 10px;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--accent);
      }}

      .reasons ul {{
        margin: 0;
        padding-left: 18px;
        color: var(--ink);
        line-height: 1.5;
      }}

      @keyframes fadeUp {{
        from {{
          transform: translateY(16px);
          opacity: 0;
        }}
        to {{
          transform: translateY(0);
          opacity: 1;
        }}
      }}

      @media (max-width: 700px) {{
        .card {{
          padding: 20px;
        }}

        .card-header {{
          flex-direction: column;
          align-items: flex-start;
        }}
      }}
    </style>
  </head>
  <body>
    <header>
      <h1>ArxivLens</h1>
      <p class="subtitle">Curated arXiv recommendations with summaries and reasons.</p>
    </header>
    <main>
      {cards_html}
    </main>
  </body>
</html>
"""


def serve_papers(papers: Iterable[object], host: str = "127.0.0.1", port: int = 0) -> str:
    with tempfile.TemporaryDirectory(prefix="arxivlens_web_") as tmp_dir:
        output_dir = Path(tmp_dir)
        html_path = output_dir / "index.html"
        html_path.write_text(_build_html(papers), encoding="utf-8")

        handler = _make_handler(str(output_dir))
        server = http.server.ThreadingHTTPServer((host, port), handler)
        url = f"http://{host}:{server.server_address[1]}/"
        print(f"Web results: {url}")
        print("Press Ctrl+C to stop the server.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return url


def _make_handler(directory: str) -> type[http.server.SimpleHTTPRequestHandler]:
    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=directory, **kwargs)

        def log_message(self, format: str, *args) -> None:
            return

    return QuietHandler
