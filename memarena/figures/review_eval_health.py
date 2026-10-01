#!/usr/bin/env python3
"""Eval Instance Health Check — shows unhealthy instances by type.

Usage:
    python3 review_eval_health.py --run MASim/runs/l_20260408_111046 --port 10004
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

RUN_DIR: Path = Path(".")


def scan_issues(run_dir: Path):
    eval_dir = run_dir / "eval_instances"
    if not eval_dir.exists():
        return {"issues": [], "summary": {}, "error": "No eval_instances directory"}

    all_issues = []
    dim_totals = {}

    for f in sorted(os.listdir(eval_dir)):
        if not f.endswith('.jsonl'):
            continue
        dim = f.replace('.jsonl', '')
        instances = []
        with open(eval_dir / f) as fh:
            for line in fh:
                instances.append(json.loads(line))
        dim_totals[dim] = len(instances)

        for i, inst in enumerate(instances):
            problems = []
            query = inst.get('query', '')
            gt = inst.get('ground_truth')
            sids = inst.get('evidence_session_ids', [])
            options = inst.get('options', inst.get('choices', []))

            # Empty query
            if not query or not query.strip():
                problems.append({'type': 'empty_query', 'severity': 'critical', 'detail': 'Query field is empty'})
            elif len(query) < 10:
                problems.append({'type': 'short_query', 'severity': 'warning', 'detail': f'Query only {len(query)} chars: "{query}"'})

            # Suspicious query text
            if query and '???' in query:
                problems.append({'type': 'placeholder_query', 'severity': 'critical', 'detail': 'Query contains ???'})

            # Missing ground truth
            if gt is None:
                problems.append({'type': 'missing_gt', 'severity': 'critical', 'detail': 'ground_truth is None'})
            elif isinstance(gt, dict) and not gt:
                problems.append({'type': 'empty_gt', 'severity': 'critical', 'detail': 'ground_truth is empty dict'})
            elif isinstance(gt, str) and not gt.strip():
                problems.append({'type': 'empty_gt', 'severity': 'critical', 'detail': 'ground_truth is empty string'})

            # Empty GT fields
            if isinstance(gt, dict):
                if dim == 'd4_permission' and 'fact' in gt and not gt.get('fact'):
                    problems.append({'type': 'empty_permission_fact', 'severity': 'high',
                                     'detail': f'Permission fact is empty (action={gt.get("expected_action")}, disclosure={gt.get("expected_disclosure")})'})
                for key in ['expected_action', 'expected_answer', 'correct_answer', 'answer']:
                    if key in gt and (gt[key] is None or (isinstance(gt[key], str) and not gt[key].strip())):
                        problems.append({'type': f'empty_gt_{key}', 'severity': 'high', 'detail': f'GT field "{key}" is empty'})

            # No evidence sessions
            if len(sids) == 0:
                problems.append({'type': 'no_evidence', 'severity': 'high',
                                 'detail': 'No evidence sessions — retrieval will return nothing'})

            # Empty options
            if isinstance(options, list) and options:
                empty_opts = [j for j, o in enumerate(options) if not str(o).strip()]
                if empty_opts:
                    problems.append({'type': 'empty_options', 'severity': 'warning',
                                     'detail': f'{len(empty_opts)} empty option(s) at indices {empty_opts}'})

            # Truncated query (ends mid-word or mid-sentence without punctuation)
            if query and len(query) > 20 and query[-1] not in '.?!"\')':
                last_word = query.split()[-1] if query.split() else ''
                if len(last_word) > 2 and not last_word[-1].isalnum():
                    pass  # ends with punctuation-like char
                elif query.count('"') % 2 != 0:
                    problems.append({'type': 'truncated_query', 'severity': 'warning',
                                     'detail': f'Query may be truncated (unmatched quotes): ...{query[-60:]}'})

            if problems:
                all_issues.append({
                    'dimension': dim,
                    'index': i,
                    'instance_id': inst.get('instance_id', ''),
                    'problems': problems,
                    'query': query[:500] if query else '',
                    'ground_truth': json.dumps(gt, ensure_ascii=False)[:500] if gt else '',
                    'n_evidence': len(sids),
                    'ego_agent': inst.get('ego_agent_id', inst.get('answerer_agent_id', '')),
                    'difficulty': inst.get('difficulty', ''),
                })

    # Summary
    type_counts = Counter()
    severity_counts = Counter()
    dim_issue_counts = Counter()
    for issue in all_issues:
        dim_issue_counts[issue['dimension']] += 1
        for p in issue['problems']:
            type_counts[p['type']] += 1
            severity_counts[p['severity']] += 1

    return {
        'issues': all_issues,
        'dim_totals': dim_totals,
        'dim_issue_counts': dict(dim_issue_counts),
        'type_counts': dict(type_counts.most_common()),
        'severity_counts': dict(severity_counts.most_common()),
        'total_instances': sum(dim_totals.values()),
        'total_issues': len(all_issues),
    }


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Eval Health Check</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }
  h1 { font-size: 24px; color: #fff; margin-bottom: 6px; }
  .subtitle { font-size: 13px; color: #6b7280; margin-bottom: 24px; }
  h2 { font-size: 16px; color: #6366f1; margin: 20px 0 10px 0; border-bottom: 1px solid #2a2d3a; padding-bottom: 6px; }

  .top-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat-card { background: #1a1d28; border: 1px solid #2a2d3a; border-radius: 10px; padding: 16px; }
  .stat-card .label { font-size: 11px; color: #9ca3af; text-transform: uppercase; }
  .stat-card .value { font-size: 28px; font-weight: 700; margin: 2px 0; }
  .stat-card .value.ok { color: #4ade80; }
  .stat-card .value.warn { color: #f59e0b; }
  .stat-card .value.bad { color: #f87171; }
  .stat-card .sub { font-size: 12px; color: #6b7280; }

  .bar-row { display: flex; align-items: center; margin: 4px 0; font-size: 13px; }
  .bar-label { width: 200px; color: #9ca3af; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; }
  .bar-track { flex: 1; height: 22px; background: #1e2030; border-radius: 4px; overflow: hidden; margin: 0 8px; position: relative; }
  .bar-fill { height: 100%; border-radius: 4px; }
  .bar-healthy { background: #059669; }
  .bar-issues { background: #dc2626; }
  .bar-count { width: 80px; text-align: right; color: #d1d5db; font-family: monospace; font-size: 12px; }

  .filter-bar { margin-bottom: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
  .filter-btn { padding: 6px 14px; border-radius: 6px; border: 1px solid #2a2d3a; background: #141620; color: #9ca3af; cursor: pointer; font-size: 12px; }
  .filter-btn.active { background: #4f46e5; color: #fff; border-color: #4f46e5; }
  .filter-btn .count { font-weight: 700; margin-left: 4px; }

  .issue-card { background: #1a1d28; border: 1px solid #2a2d3a; border-radius: 10px; padding: 16px; margin-bottom: 10px; }
  .issue-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .issue-dim { font-size: 12px; font-weight: 600; padding: 2px 10px; border-radius: 4px; }
  .issue-id { font-size: 12px; color: #6b7280; font-family: monospace; }
  .issue-badges { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 8px; }
  .badge { font-size: 11px; padding: 3px 10px; border-radius: 4px; font-weight: 600; }
  .badge.critical { background: #450a0a; color: #fca5a5; }
  .badge.high { background: #78350f; color: #fbbf24; }
  .badge.warning { background: #1e1b4b; color: #a5b4fc; }
  .issue-query { font-size: 13px; color: #d1d5db; background: #141620; padding: 10px; border-radius: 6px; margin: 6px 0; line-height: 1.5; max-height: 100px; overflow-y: auto; }
  .issue-gt { font-size: 12px; color: #9ca3af; background: #141620; padding: 8px; border-radius: 6px; margin: 4px 0; font-family: monospace; max-height: 80px; overflow-y: auto; word-break: break-all; }
  .issue-meta { font-size: 12px; color: #6b7280; margin-top: 6px; }

  .dim-d1_conflict { background: #059669; color: #fff; }
  .dim-d2_anaphora { background: #0891b2; color: #fff; }
  .dim-d3_confabulation { background: #7c3aed; color: #fff; }
  .dim-d4_permission { background: #dc2626; color: #fff; }
  .dim-d5_cloze { background: #ca8a04; color: #fff; }
  .dim-d6_metadata { background: #0d9488; color: #fff; }
  .dim-d7_qa { background: #2563eb; color: #fff; }
  .dim-d8_temporal { background: #c026d3; color: #fff; }
  .dim-d9_negation { background: #e11d48; color: #fff; }
  .dim-d10_counterfactual { background: #ea580c; color: #fff; }
  .dim-d11_exception { background: #4f46e5; color: #fff; }
</style>
</head>
<body>

<h1>Eval Instance Health Check</h1>
<div class="subtitle" id="subtitle"></div>
<div id="content"><p style="color:#9ca3af;">Loading...</p></div>

<script>
let data = {};
let activeFilter = 'all';

async function load() {
  const resp = await fetch('/api/health');
  data = await resp.json();
  render();
}

function setFilter(f) {
  activeFilter = f;
  render();
}

function render() {
  const d = data;
  const issues = d.issues || [];
  const healthy = d.total_instances - d.total_issues;
  const pct = ((healthy / d.total_instances) * 100).toFixed(1);

  document.getElementById('subtitle').textContent =
    `${d.total_instances} total instances, ${d.total_issues} with issues (${pct}% healthy)`;

  // Top cards
  let html = `<div class="top-grid">
    <div class="stat-card"><div class="label">Total Instances</div><div class="value" style="color:#fff;">${d.total_instances}</div></div>
    <div class="stat-card"><div class="label">Healthy</div><div class="value ok">${healthy}</div><div class="sub">${pct}%</div></div>
    <div class="stat-card"><div class="label">Issues</div><div class="value ${d.total_issues > 50 ? 'bad' : 'warn'}">${d.total_issues}</div></div>
    <div class="stat-card"><div class="label">Critical</div><div class="value bad">${d.severity_counts['critical'] || 0}</div></div>
    <div class="stat-card"><div class="label">High</div><div class="value warn">${d.severity_counts['high'] || 0}</div></div>
    <div class="stat-card"><div class="label">Warning</div><div class="value" style="color:#a5b4fc;">${d.severity_counts['warning'] || 0}</div></div>
  </div>`;

  // Per-dimension health bar
  html += '<h2>Per-Dimension Health</h2>';
  const dims = Object.keys(d.dim_totals || {}).sort();
  const maxDim = Math.max(...Object.values(d.dim_totals || {}));
  for (const dim of dims) {
    const total = d.dim_totals[dim];
    const bad = d.dim_issue_counts[dim] || 0;
    const good = total - bad;
    const goodPct = (good / total * 100).toFixed(0);
    const badPct = (bad / total * 100).toFixed(0);
    html += `<div class="bar-row">
      <span class="bar-label">${dim}</span>
      <div class="bar-track" style="max-width:${total/maxDim*100}%;">
        <div class="bar-fill bar-healthy" style="width:${goodPct}%;display:inline-block;"></div>
        <div class="bar-fill bar-issues" style="width:${badPct}%;display:inline-block;"></div>
      </div>
      <span class="bar-count">${good}/${total} ${bad > 0 ? '('+bad+' bad)' : ''}</span>
    </div>`;
  }

  // Issue type breakdown
  html += '<h2>Issue Types</h2>';
  const types = d.type_counts || {};
  const maxType = Math.max(...Object.values(types));
  for (const [type, count] of Object.entries(types)) {
    html += `<div class="bar-row">
      <span class="bar-label">${type}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${count/maxType*100}%;background:#f59e0b;"></div></div>
      <span class="bar-count">${count}</span>
    </div>`;
  }

  // Filter buttons
  const typeCounts = {};
  const dimCounts = {};
  for (const issue of issues) {
    dimCounts[issue.dimension] = (dimCounts[issue.dimension] || 0) + 1;
    for (const p of issue.problems) {
      typeCounts[p.type] = (typeCounts[p.type] || 0) + 1;
    }
  }

  html += '<h2>All Issues</h2><div class="filter-bar">';
  html += `<div class="filter-btn ${activeFilter==='all'?'active':''}" onclick="setFilter('all')">All<span class="count">${issues.length}</span></div>`;
  for (const [type, count] of Object.entries(typeCounts).sort((a,b) => b[1]-a[1])) {
    html += `<div class="filter-btn ${activeFilter===type?'active':''}" onclick="setFilter('${type}')">${type}<span class="count">${count}</span></div>`;
  }
  html += '</div>';

  // Issue cards
  const filtered = activeFilter === 'all' ? issues : issues.filter(i => i.problems.some(p => p.type === activeFilter));
  for (const issue of filtered.slice(0, 100)) {
    html += `<div class="issue-card">
      <div class="issue-header">
        <span class="issue-dim dim-${issue.dimension}">${issue.dimension}</span>
        <span class="issue-id">${issue.instance_id} (#${issue.index})</span>
      </div>
      <div class="issue-badges">
        ${issue.problems.map(p => `<span class="badge ${p.severity}">${p.type}: ${p.detail}</span>`).join('')}
      </div>
      ${issue.query ? `<div class="issue-query">${issue.query}</div>` : ''}
      ${issue.ground_truth ? `<div class="issue-gt">${issue.ground_truth}</div>` : ''}
      <div class="issue-meta">Agent: ${issue.ego_agent} · Evidence: ${issue.n_evidence} sessions · Difficulty: ${issue.difficulty}</div>
    </div>`;
  }
  if (filtered.length > 100) html += `<p style="color:#6b7280;">Showing first 100 of ${filtered.length}</p>`;

  document.getElementById('content').innerHTML = html;
}

load();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[eval-health] {args[0]} {args[1]}\n")

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html, status=200):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._html_response(HTML_PAGE)
        elif path == "/api/health":
            self._json_response(scan_issues(RUN_DIR))
        else:
            self.send_error(404)


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser(description="Eval Instance Health Check")
    parser.add_argument("--run", "-r", required=True)
    parser.add_argument("--port", "-p", type=int, default=10004)
    args = parser.parse_args()

    RUN_DIR = Path(args.run)
    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Serving on http://0.0.0.0:{args.port}")
    print(f"Open http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
