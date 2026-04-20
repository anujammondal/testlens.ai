"""
Generate HTML Report for Duplicate Test Cases

Creates an HTML report based on duplicate detection via:
- normalized_summary (exact match)
- embedding (cosine similarity)
"""

import html
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from findDuplicateTestCases import DuplicateTestCaseFinder, DuplicateGroup


def _escape(s: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(s)) if s else ""


def _step_details_html(step_details: dict) -> str:
    """Render stepDetails as HTML list."""
    if not step_details:
        return "<em>No steps</em>"
    keys = sorted(
        step_details.keys(),
        key=lambda k: int(k.split("_")[1]) if "_" in k and k.split("_")[1].isdigit() else 0,
    )
    items = []
    for k in keys:
        v = step_details.get(k, "")
        if v:
            items.append(f'<li><span class="step-num">{_escape(k)}</span> {_escape(v)}</li>')
    return f"<ol>{''.join(items)}</ol>" if items else "<em>No steps</em>"


def _testcase_card(tc: dict) -> str:
    """Render a single test case as an HTML card."""
    key = tc.get("key", tc.get("id", "—"))
    summary = tc.get("summary", "")
    normalized = tc.get("normalized_summary", "")
    folder_name = ""
    if tc.get("folder"):
        folder_name = tc["folder"].get("name", "") or ""
    priority = tc.get("priority", "")
    steps_html = _step_details_html(tc.get("stepDetails", {}))

    return f"""
    <div class="tc-card">
      <div class="tc-header">
        <span class="tc-key">{_escape(key)}</span>
        <span class="tc-folder">{_escape(folder_name)}</span>
        <span class="tc-priority">{_escape(priority)}</span>
      </div>
      <div class="tc-body">
        <p class="tc-summary"><strong>Summary:</strong> {_escape(summary)}</p>
        <p class="tc-normalized"><strong>Normalized:</strong> {_escape(normalized)}</p>
        <details class="tc-steps">
          <summary>Step details</summary>
          <div class="steps-content">{steps_html}</div>
        </details>
      </div>
    </div>
    """


def _duplicate_group_html(group: DuplicateGroup, section: str) -> str:
    """Render a duplicate group as HTML."""
    score_html = ""
    if group.similarity_score is not None and section == "embedding":
        score_pct = group.similarity_score * 100
        score_class = "high" if score_pct >= 95 else "medium" if score_pct >= 92 else "low"
        score_html = f'<span class="sim-badge {score_class}">{score_pct:.1f}% similar</span>'

    cards = "".join(_testcase_card(tc) for tc in group.testcases)
    keys = [tc.get("key", tc.get("id", "?")) for tc in group.testcases]

    return f"""
    <div class="dup-group">
      <div class="group-header">
        <span class="group-ids">{_escape(", ".join(keys))}</span>
        {score_html}
      </div>
      <div class="group-cards">
        {cards}
      </div>
    </div>
    """


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Duplicate Test Cases Report - TestLens</title>
  <style>
    :root {
      --bg: #0f1419;
      --surface: #1a2332;
      --surface-hover: #232f42;
      --border: #2d3a4f;
      --text: #e6edf3;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --accent-dim: #388bfd66;
      --success: #3fb950;
      --warning: #d29922;
      --danger: #f85149;
    }
    * { box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 2rem;
      line-height: 1.5;
    }
    .container { max-width: 960px; margin: 0 auto; }
    h1 { font-size: 1.75rem; font-weight: 600; margin-bottom: 0.5rem; }
    .meta { color: var(--text-muted); font-size: 0.9rem; margin-bottom: 2rem; }
    .stats {
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;
      margin-bottom: 2rem;
      padding: 1rem 1.25rem;
      background: var(--surface);
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    .stat { display: flex; flex-direction: column; gap: 0.25rem; }
    .stat-value { font-size: 1.5rem; font-weight: 600; color: var(--accent); }
    .stat-label { font-size: 0.8rem; color: var(--text-muted); }
    section {
      margin-bottom: 2.5rem;
      padding: 1.25rem;
      background: var(--surface);
      border-radius: 8px;
      border: 1px solid var(--border);
    }
    section h2 {
      font-size: 1.15rem;
      font-weight: 600;
      margin: 0 0 1rem 0;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .dup-group {
      margin-bottom: 1.5rem;
      padding: 1rem;
      background: var(--surface-hover);
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .dup-group:last-child { margin-bottom: 0; }
    .group-header {
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-bottom: 1rem;
    }
    .group-ids { font-weight: 600; color: var(--accent); }
    .sim-badge {
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 500;
    }
    .sim-badge.high { background: var(--success); color: #fff; }
    .sim-badge.medium { background: var(--warning); color: #0d1117; }
    .sim-badge.low { background: var(--danger); color: #fff; }
    .group-cards { display: flex; flex-direction: column; gap: 1rem; }
    .tc-card {
      padding: 1rem;
      background: var(--bg);
      border-radius: 6px;
      border: 1px solid var(--border);
    }
    .tc-header {
      display: flex;
      align-items: center;
      gap: 1rem;
      margin-bottom: 0.75rem;
    }
    .tc-key { font-weight: 600; color: var(--accent); }
    .tc-folder { font-size: 0.85rem; color: var(--text-muted); }
    .tc-priority {
      font-size: 0.75rem;
      padding: 0.15rem 0.4rem;
      background: var(--accent-dim);
      border-radius: 4px;
    }
    .tc-body p { margin: 0.5rem 0; font-size: 0.9rem; }
    .tc-summary { color: var(--text); }
    .tc-normalized { color: var(--text-muted); font-size: 0.85rem; }
    details.tc-steps { margin-top: 0.75rem; }
    summary { cursor: pointer; font-size: 0.85rem; color: var(--text-muted); }
    .steps-content { margin-top: 0.5rem; padding-left: 1.25rem; }
    .steps-content ol { margin: 0; padding-left: 1rem; }
    .steps-content li { margin: 0.35rem 0; font-size: 0.85rem; }
    .step-num { font-weight: 500; color: var(--accent); margin-right: 0.25rem; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Duplicate Test Cases Report</h1>
    <p class="meta">Generated on {generated_at} · TestLens AI</p>

    <div class="stats">
      <div class="stat">
        <span class="stat-value">{total_tcs}</span>
        <span class="stat-label">Total test cases</span>
      </div>
      <div class="stat">
        <span class="stat-value">{norm_groups}</span>
        <span class="stat-label">Duplicate groups (normalized summary)</span>
      </div>
      <div class="stat">
        <span class="stat-value">{emb_groups}</span>
        <span class="stat-label">Duplicate groups (embedding similarity)</span>
      </div>
      <div class="stat">
        <span class="stat-value">{dup_tcs}</span>
        <span class="stat-label">Test cases in duplicate groups</span>
      </div>
    </div>

    <section>
      <h2>📝 By Normalized Summary</h2>
      <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem;">
        Test cases with identical normalized summaries (same semantic goal).
      </p>
      {normalized_section}
    </section>

    <section>
      <h2>🔢 By Embedding Similarity</h2>
      <p style="color: var(--text-muted); font-size: 0.9rem; margin-bottom: 1rem;">
        Test cases with embedding cosine similarity ≥ {threshold}% (semantic near-duplicates).
      </p>
      {embedding_section}
    </section>
  </div>
</body>
</html>
"""


def generate_report(
    embeddings_file: str = "qmetry_testcases_embeddings.json",
    output_file: str = "reports/duplicate_testcases_report.html",
    project_root: Optional[Path] = None,
    threshold: float = 0.92,
) -> Path:
    """
    Generate HTML report for duplicate test cases (normalized_summary + embedding).

    Returns the path to the generated HTML file.
    """
    project_root = project_root or Path(__file__).resolve().parent.parent.parent
    output_path = project_root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)

    finder = DuplicateTestCaseFinder(
        embeddings_file=embeddings_file,
        project_root=project_root,
        embedding_similarity_threshold=threshold,
    )
    finder.load()

    norm_groups = finder.find_by_normalized_summary()
    emb_groups = finder.find_by_embedding()

    # Unique test case count in duplicates (approximate - may overlap)
    dup_ids = set()
    for g in norm_groups + emb_groups:
        for tc in g.testcases:
            dup_ids.add(tc.get("id") or tc.get("key", ""))

    norm_html = "".join(_duplicate_group_html(g, "normalized_summary") for g in norm_groups)
    if not norm_html:
        norm_html = '<p style="color: var(--text-muted);">No duplicates found by normalized summary.</p>'

    emb_html = "".join(_duplicate_group_html(g, "embedding") for g in emb_groups)
    if not emb_html:
        emb_html = '<p style="color: var(--text-muted);">No duplicates found by embedding similarity.</p>'

    html_content = HTML_TEMPLATE.replace("{generated_at}", datetime.now().strftime("%Y-%m-%d %H:%M"))
    html_content = html_content.replace("{total_tcs}", str(len(finder.testcases)))
    html_content = html_content.replace("{norm_groups}", str(len(norm_groups)))
    html_content = html_content.replace("{emb_groups}", str(len(emb_groups)))
    html_content = html_content.replace("{dup_tcs}", str(len(dup_ids)))
    html_content = html_content.replace("{threshold}", f"{threshold * 100:.0f}")
    html_content = html_content.replace("{normalized_section}", norm_html)
    html_content = html_content.replace("{embedding_section}", emb_html)

    output_path.write_text(html_content, encoding="utf-8")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML report for duplicate test cases")
    parser.add_argument(
        "--file", "-f",
        default="qmetry_testcases_embeddings.json",
        help="Path to embeddings JSON",
    )
    parser.add_argument(
        "--output", "-o",
        default="reports/duplicate_testcases_report.html",
        help="Output HTML file path",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.92,
        help="Embedding similarity threshold",
    )
    args = parser.parse_args()

    path = generate_report(
        embeddings_file=args.file,
        output_file=args.output,
        threshold=args.threshold,
    )
    print(f"✅ Report generated: {path}")
