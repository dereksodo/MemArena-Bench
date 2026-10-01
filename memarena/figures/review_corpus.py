#!/usr/bin/env python3
"""Corpus Viewer — per-day breakdown of generated dialogue sessions.

Auto-refreshes every 60s to show progress during generation.

Usage:
    python3 review_corpus.py --run MASim/runs/l_20260408_111046 --port 10003
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

RUN_DIR: Path = Path(".")


def compute_corpus_stats(run_dir: Path):
    """Read corpus_sessions.jsonl and compute per-day stats. Re-reads on each call for live updates."""
    corpus_path = run_dir / "corpus_sessions.jsonl"
    if not corpus_path.exists():
        return {"days": [], "totals": {}, "error": "corpus_sessions.jsonl not found"}

    sessions = []
    with open(corpus_path) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))

    if not sessions:
        return {"days": [], "totals": {}, "error": "No sessions yet"}

    # Group by day
    by_day = defaultdict(list)
    for s in sessions:
        day = int(s.get("start_time", 0))
        by_day[day].append(s)

    # Per-agent stats
    agent_tokens = Counter()
    agent_sessions = Counter()

    days = []
    total_sessions = 0
    total_turns = 0
    total_words = 0
    total_pp = 0
    total_pa = 0

    for day in sorted(by_day.keys()):
        ss = by_day[day]
        turns = sum(len(s.get("turns", [])) for s in ss)
        words = sum(len(t.get("text", "").split()) for s in ss for t in s.get("turns", []))
        tokens = int(words * 1.3)

        # PP vs PA
        pp = sum(1 for s in ss if s.get("session_type", "") != "person_agent"
                 and not s.get("session_id", "").startswith("pa_"))
        pa = len(ss) - pp

        # Modality
        modalities = Counter(s.get("modality", "unknown") for s in ss)

        # Session types
        types = Counter(s.get("session_type", "unknown") for s in ss)

        # Interest domains
        domains = Counter(s.get("interest_domain", "") for s in ss)
        domains.pop("", None)

        # Per-agent token tracking
        for s in ss:
            for p in s.get("participants", []):
                w = sum(len(t.get("text", "").split()) for t in s.get("turns", []))
                agent_tokens[p] += int(w * 1.3 / max(len(s.get("participants", [])), 1))
                agent_sessions[p] += 1

        # Words per turn distribution
        turn_lengths = [len(t.get("text", "").split()) for s in ss for t in s.get("turns", [])]
        avg_words = round(sum(turn_lengths) / len(turn_lengths), 1) if turn_lengths else 0
        short_turns = sum(1 for w in turn_lengths if w < 20)
        medium_turns = sum(1 for w in turn_lengths if 20 <= w < 80)
        long_turns = sum(1 for w in turn_lengths if w >= 80)

        # Sample conversations (first 3 PP sessions with turns)
        samples = []
        for s in ss:
            if len(samples) >= 2:
                break
            if s.get("turns") and len(s["turns"]) >= 4:
                sample_turns = []
                for t in s["turns"][:6]:
                    speaker = t.get("speaker_id", "?").replace("_", " ").title()
                    text = t.get("text", "")
                    if len(text) > 300:
                        text = text[:300] + "..."
                    sample_turns.append({"speaker": speaker, "text": text})
                samples.append({
                    "participants": [p.replace("_", " ").title() for p in s.get("participants", [])],
                    "turns": sample_turns,
                    "domain": s.get("interest_domain", ""),
                    "type": s.get("session_type", ""),
                })

        days.append({
            "day": day,
            "n_sessions": len(ss),
            "n_pp": pp,
            "n_pa": pa,
            "n_turns": turns,
            "n_words": words,
            "n_tokens": tokens,
            "avg_turns_per_session": round(turns / len(ss), 1) if ss else 0,
            "avg_words_per_turn": avg_words,
            "short_turns": short_turns,
            "medium_turns": medium_turns,
            "long_turns": long_turns,
            "modalities": dict(modalities.most_common()),
            "session_types": dict(types.most_common()),
            "top_domains": dict(domains.most_common(5)),
            "samples": samples,
        })

        total_sessions += len(ss)
        total_turns += turns
        total_words += words
        total_pp += pp
        total_pa += pa

    n_agents = len(agent_tokens)
    n_days_done = len(days)

    # Per-agent summary (top/bottom 5)
    agent_list = sorted(agent_tokens.items(), key=lambda x: -x[1])
    top_agents = [{"name": a.replace("_", " ").title(), "tokens": t, "sessions": agent_sessions[a]}
                  for a, t in agent_list[:5]]
    bottom_agents = [{"name": a.replace("_", " ").title(), "tokens": t, "sessions": agent_sessions[a]}
                     for a, t in agent_list[-5:]]

    totals = {
        "n_sessions": total_sessions,
        "n_pp": total_pp,
        "n_pa": total_pa,
        "n_turns": total_turns,
        "n_words": total_words,
        "n_tokens": int(total_words * 1.3),
        "n_agents": n_agents,
        "n_days_done": n_days_done,
        "tokens_per_person_day": int(total_words * 1.3 / max(n_agents, 1) / max(n_days_done, 1)),
        "avg_words_per_turn": round(total_words / max(total_turns, 1), 1),
        "top_agents": top_agents,
        "bottom_agents": bottom_agents,
    }

    return {"days": days, "totals": totals}


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Corpus Viewer — By Day</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }
  h1 { font-size: 24px; color: #fff; margin-bottom: 6px; }
  .subtitle { font-size: 13px; color: #6b7280; margin-bottom: 24px; }
  h2 { font-size: 16px; color: #6366f1; margin: 20px 0 10px 0; }

  .top-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat-card { background: #1a1d28; border: 1px solid #2a2d3a; border-radius: 10px; padding: 16px; }
  .stat-card .label { font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
  .stat-card .value { font-size: 28px; color: #fff; font-weight: 700; margin: 2px 0; }
  .stat-card .sub { font-size: 12px; color: #6b7280; }
  .stat-card .target { font-size: 12px; color: #4ade80; }
  .stat-card .warn { font-size: 12px; color: #f59e0b; }

  .day-card { background: #1a1d28; border: 1px solid #2a2d3a; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  .day-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
  .day-header h3 { font-size: 18px; color: #fff; }
  .day-header .tokens { font-size: 14px; color: #4ade80; font-weight: 600; }

  .day-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-bottom: 12px; }
  .day-stat { background: #141620; border-radius: 8px; padding: 10px; text-align: center; }
  .day-stat .ds-val { font-size: 20px; font-weight: 700; color: #fff; }
  .day-stat .ds-lbl { font-size: 11px; color: #9ca3af; }

  .turn-dist { display: flex; gap: 4px; height: 20px; border-radius: 4px; overflow: hidden; margin: 8px 0; }
  .turn-dist .seg { height: 100%; transition: width 0.3s; }
  .seg-short { background: #f59e0b; }
  .seg-medium { background: #059669; }
  .seg-long { background: #6366f1; }
  .turn-legend { display: flex; gap: 16px; font-size: 11px; color: #9ca3af; }
  .turn-legend span::before { content: ''; display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
  .legend-short::before { background: #f59e0b; }
  .legend-medium::before { background: #059669; }
  .legend-long::before { background: #6366f1; }

  .tags { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
  .tag { font-size: 11px; padding: 3px 10px; border-radius: 6px; background: #1e2030; color: #d1d5db; }

  .sample { background: #141620; border: 1px solid #1e2030; border-radius: 8px; padding: 12px; margin-top: 10px; font-size: 13px; }
  .sample-header { font-size: 12px; color: #6366f1; margin-bottom: 8px; font-weight: 600; }
  .sample-turn { margin: 4px 0; line-height: 1.5; }
  .sample-turn .speaker { color: #a78bfa; font-weight: 600; }
  .sample-turn .text { color: #d1d5db; }
  .sample-toggle { font-size: 12px; color: #6366f1; cursor: pointer; margin-top: 8px; }

  .agent-table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .agent-table th { text-align: left; color: #9ca3af; font-weight: 600; padding: 6px 12px; border-bottom: 1px solid #2a2d3a; }
  .agent-table td { padding: 6px 12px; border-bottom: 1px solid #1e2030; }

  .bar-inline { display: inline-block; height: 12px; border-radius: 3px; vertical-align: middle; margin-left: 8px; }
</style>
</head>
<body>

<h1>Corpus Viewer</h1>
<div class="subtitle" id="refreshNote">Auto-refreshes every 60s during generation</div>

<div id="content"><p style="color:#9ca3af;">Loading...</p></div>

<script>
let data = {};

async function load() {
  const resp = await fetch('/api/corpus');
  data = await resp.json();
  render();
}

function render() {
  const t = data.totals || {};
  const days = data.days || [];

  if (data.error) {
    document.getElementById('content').innerHTML = `<div class="stat-card"><p style="color:#f59e0b;">${data.error}</p></div>`;
    return;
  }

  const tpd = t.tokens_per_person_day || 0;
  const tpdClass = tpd >= 9000 ? 'target' : 'warn';
  const tpdLabel = tpd >= 9000 ? 'On target' : 'Below 10K target';

  let html = `
    <div class="top-grid">
      <div class="stat-card">
        <div class="label">Days Complete</div>
        <div class="value">${t.n_days_done || 0} / 15</div>
        <div class="sub">${Math.round((t.n_days_done||0)/15*100)}%</div>
      </div>
      <div class="stat-card">
        <div class="label">Total Sessions</div>
        <div class="value">${(t.n_sessions||0).toLocaleString()}</div>
        <div class="sub">${(t.n_pp||0).toLocaleString()} PP + ${(t.n_pa||0).toLocaleString()} PA</div>
      </div>
      <div class="stat-card">
        <div class="label">Total Turns</div>
        <div class="value">${(t.n_turns||0).toLocaleString()}</div>
        <div class="sub">${t.avg_words_per_turn || 0} words/turn avg</div>
      </div>
      <div class="stat-card">
        <div class="label">Total Tokens</div>
        <div class="value">${((t.n_tokens||0)/1000000).toFixed(2)}M</div>
        <div class="sub">${(t.n_tokens||0).toLocaleString()}</div>
      </div>
      <div class="stat-card">
        <div class="label">Tok/Person/Day</div>
        <div class="value">${tpd.toLocaleString()}</div>
        <div class="${tpdClass}">${tpdLabel}</div>
      </div>
      <div class="stat-card">
        <div class="label">Agents</div>
        <div class="value">${t.n_agents || 0}</div>
      </div>
    </div>
  `;

  // Agent distribution
  if (t.top_agents && t.top_agents.length) {
    const maxTok = Math.max(...t.top_agents.map(a => a.tokens));
    html += `<h2>Token Distribution (Top / Bottom 5 Agents)</h2>
      <table class="agent-table">
        <tr><th>Agent</th><th>Tokens</th><th>Sessions</th><th></th></tr>
        ${t.top_agents.map(a => `<tr><td>${a.name}</td><td>${a.tokens.toLocaleString()}</td><td>${a.sessions}</td>
          <td><div class="bar-inline" style="width:${Math.round(a.tokens/maxTok*120)}px;background:#4ade80;"></div></td></tr>`).join('')}
        <tr><td colspan="4" style="color:#4b5563;font-size:11px;">...</td></tr>
        ${(t.bottom_agents||[]).map(a => `<tr><td>${a.name}</td><td>${a.tokens.toLocaleString()}</td><td>${a.sessions}</td>
          <td><div class="bar-inline" style="width:${Math.round(a.tokens/maxTok*120)}px;background:#f59e0b;"></div></td></tr>`).join('')}
      </table>`;
  }

  // Per-day cards
  html += `<h2>Per-Day Breakdown</h2>`;
  for (const d of days) {
    const totalTurns = d.short_turns + d.medium_turns + d.long_turns;
    const sPct = totalTurns ? (d.short_turns/totalTurns*100).toFixed(0) : 0;
    const mPct = totalTurns ? (d.medium_turns/totalTurns*100).toFixed(0) : 0;
    const lPct = totalTurns ? (d.long_turns/totalTurns*100).toFixed(0) : 0;

    html += `
      <div class="day-card">
        <div class="day-header">
          <h3>Day ${d.day}</h3>
          <span class="tokens">${d.n_tokens.toLocaleString()} tokens</span>
        </div>
        <div class="day-stats">
          <div class="day-stat"><div class="ds-val">${d.n_sessions}</div><div class="ds-lbl">sessions</div></div>
          <div class="day-stat"><div class="ds-val">${d.n_pp}</div><div class="ds-lbl">PP</div></div>
          <div class="day-stat"><div class="ds-val">${d.n_pa}</div><div class="ds-lbl">PA</div></div>
          <div class="day-stat"><div class="ds-val">${d.n_turns}</div><div class="ds-lbl">turns</div></div>
          <div class="day-stat"><div class="ds-val">${d.avg_turns_per_session}</div><div class="ds-lbl">turns/sess</div></div>
          <div class="day-stat"><div class="ds-val">${d.avg_words_per_turn}</div><div class="ds-lbl">words/turn</div></div>
        </div>

        <div class="turn-dist">
          <div class="seg seg-short" style="width:${sPct}%;" title="<20 words: ${d.short_turns}"></div>
          <div class="seg seg-medium" style="width:${mPct}%;" title="20-80 words: ${d.medium_turns}"></div>
          <div class="seg seg-long" style="width:${lPct}%;" title="80+ words: ${d.long_turns}"></div>
        </div>
        <div class="turn-legend">
          <span class="legend-short">&lt;20w (${sPct}%)</span>
          <span class="legend-medium">20-80w (${mPct}%)</span>
          <span class="legend-long">80+w (${lPct}%)</span>
        </div>

        <div class="tags">
          ${Object.entries(d.session_types).map(([k,v]) => `<span class="tag">${k}: ${v}</span>`).join('')}
        </div>
        <div class="tags">
          ${Object.entries(d.top_domains).map(([k,v]) => `<span class="tag">${k}: ${v}</span>`).join('')}
        </div>

        ${d.samples.map((s,i) => `
          <div class="sample" id="sample-${d.day}-${i}">
            <div class="sample-header">${s.participants.join(' & ')} — ${s.type} (${s.domain})</div>
            ${s.turns.map(t => `<div class="sample-turn"><span class="speaker">${t.speaker}:</span> <span class="text">${t.text}</span></div>`).join('')}
          </div>
        `).join('')}
      </div>
    `;
  }

  document.getElementById('content').innerHTML = html;
}

load();
setInterval(load, 60000);
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[corpus] {args[0]} {args[1]}\n")

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
        elif path == "/api/corpus":
            self._json_response(compute_corpus_stats(RUN_DIR))
        else:
            self.send_error(404)


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser(description="Corpus Viewer — Per-Day Breakdown")
    parser.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")
    parser.add_argument("--port", "-p", type=int, default=10003)
    args = parser.parse_args()

    RUN_DIR = Path(args.run)
    print(f"Serving corpus viewer for {RUN_DIR}")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"Open http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
