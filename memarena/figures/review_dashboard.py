#!/usr/bin/env python3
"""MemArena-L Dashboard — overview of personas, schedules, events, and corpus stats.

Usage:
    python3 review_dashboard.py --run MASim/runs/l_20260408_111046 --port 10002
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

RUN_DIR: Path = Path(".")
DATA: dict = {}


def load_data(run_dir: Path):
    global DATA
    DATA = {"run_dir": str(run_dir)}

    # Personas
    personas = []
    p_path = run_dir / "agents_personas.jsonl"
    if p_path.exists():
        with open(p_path) as f:
            for line in f:
                personas.append(json.loads(line))
    DATA["personas"] = personas

    # Schedules
    schedules = []
    s_path = run_dir / "agent_schedules.jsonl"
    if s_path.exists():
        with open(s_path) as f:
            for line in f:
                schedules.append(json.loads(line))
    DATA["schedules"] = schedules

    # Events
    events = []
    e_path = run_dir / "events.jsonl"
    if e_path.exists():
        with open(e_path) as f:
            for line in f:
                events.append(json.loads(line))
    DATA["events"] = events

    # Corpus sessions
    sessions = []
    c_path = run_dir / "corpus_sessions.jsonl"
    if c_path.exists():
        with open(c_path) as f:
            for line in f:
                sessions.append(json.loads(line))
    DATA["sessions"] = sessions

    # Locations
    loc_path = run_dir / "locations.json"
    DATA["locations"] = json.loads(loc_path.read_text()) if loc_path.exists() else []

    # Groups
    grp_path = run_dir / "groups.json"
    DATA["groups"] = json.loads(grp_path.read_text()) if grp_path.exists() else []

    # Eval instances
    eval_dir = run_dir / "eval_instances"
    eval_counts = {}
    if eval_dir.exists():
        for f in eval_dir.glob("*.jsonl"):
            with open(f) as fh:
                eval_counts[f.stem] = sum(1 for _ in fh)
    DATA["eval_counts"] = eval_counts

    # Review statuses
    for name in ["review_status.json", "schedule_review_status.json"]:
        p = run_dir / name
        if p.exists():
            DATA[name.replace(".json", "")] = json.loads(p.read_text())


def compute_stats():
    """Compute all dashboard statistics."""
    stats = {}
    personas = DATA.get("personas", [])
    schedules = DATA.get("schedules", [])
    events = DATA.get("events", [])
    sessions = DATA.get("sessions", [])
    locations = DATA.get("locations", [])
    groups = DATA.get("groups", [])

    # Persona stats
    stats["n_agents"] = len(personas)
    if personas:
        ages = [p["persona"]["age"] for p in personas]
        stats["age_range"] = f"{min(ages)}-{max(ages)}"
        stats["age_avg"] = round(sum(ages) / len(ages), 1)
        stats["occupations"] = [{"name": p["persona"]["name"], "occupation": p["persona"]["occupation"]} for p in personas]
        genders = Counter(p["persona"].get("demographics", {}).get("gender", "?") for p in personas)
        stats["gender_dist"] = dict(genders.most_common())
        edu = Counter(p["persona"].get("education_level", "?") for p in personas)
        stats["edu_dist"] = dict(edu.most_common())
        comm = Counter(p["persona"].get("communication_style", "?") for p in personas)
        stats["comm_dist"] = dict(comm.most_common())

    # Persona review status
    review = DATA.get("review_status", {})
    stats["personas_approved"] = sum(1 for v in review.values() if v == "approved")
    stats["personas_pending"] = sum(1 for v in review.values() if v == "pending")

    # Schedule stats
    stats["n_schedules"] = len(schedules)
    if schedules:
        total_entries = sum(len(s["entries"]) for s in schedules)
        stats["total_schedule_entries"] = total_entries
        stats["avg_entries_per_agent"] = round(total_entries / len(schedules), 1)
        type_counts = Counter()
        for s in schedules:
            for e in s["entries"]:
                type_counts[e["activity_type"]] += 1
        stats["schedule_type_dist"] = dict(type_counts.most_common())

    sched_review = DATA.get("schedule_review_status", {})
    stats["schedules_approved"] = sum(1 for v in sched_review.values() if v == "approved")
    stats["schedules_pending"] = sum(1 for v in sched_review.values() if v == "pending")

    # Location stats
    stats["n_locations"] = len(locations)
    if locations:
        loc_types = Counter(l.get("location_type", "?") for l in locations)
        stats["location_type_dist"] = dict(loc_types.most_common())

    # Group stats
    stats["n_groups"] = len(groups)
    if groups:
        grp_types = Counter(g.get("group_type", "?") for g in groups)
        stats["group_type_dist"] = dict(grp_types.most_common())

    # Event stats
    stats["n_events"] = len(events)
    if events:
        cat_counts = Counter(e.get("category", "?") for e in events)
        stats["event_category_dist"] = dict(cat_counts.most_common())
        domain_counts = Counter(e.get("interest_domain", "?") for e in events)
        stats["event_domain_dist"] = dict(domain_counts.most_common())
        vis_sizes = [len(e.get("visibility_mask", [])) for e in events]
        stats["event_avg_visibility"] = round(sum(vis_sizes) / len(vis_sizes), 1)
        # Events per day
        day_counts = Counter(int(e["timestamp"]) for e in events)
        stats["events_per_day"] = dict(sorted(day_counts.items()))
        # Event list for detail view
        stats["event_list"] = [{
            "event_id": e["event_id"],
            "timestamp": e["timestamp"],
            "day": int(e["timestamp"]) + 1,
            "event_type": e.get("event_type", ""),
            "content": e.get("content", ""),
            "category": e.get("category", ""),
            "interest_domain": e.get("interest_domain", ""),
            "visibility_count": len(e.get("visibility_mask", [])),
            "visibility_mask": e.get("visibility_mask", []),
            "location_id": e.get("location_id", ""),
        } for e in events]

    # Corpus stats
    stats["n_sessions"] = len(sessions)
    if sessions:
        total_turns = sum(len(s.get("turns", [])) for s in sessions)
        total_words = sum(len(t.get("text", "").split()) for s in sessions for t in s.get("turns", []))
        stats["total_turns"] = total_turns
        stats["total_words"] = total_words
        stats["total_tokens_est"] = int(total_words * 1.3)
        stats["avg_turns_per_session"] = round(total_turns / len(sessions), 1)
        stats["avg_words_per_turn"] = round(total_words / total_turns, 1) if total_turns else 0
        n_days = 15
        stats["tokens_per_person_day"] = int(total_words * 1.3 / max(len(personas), 1) / n_days)

        modality_counts = Counter(s.get("modality", "?") for s in sessions)
        stats["modality_dist"] = dict(modality_counts.most_common())

        sess_type = Counter(s.get("session_type", "?") for s in sessions)
        stats["session_type_dist"] = dict(sess_type.most_common())

        domain_counts = Counter(s.get("interest_domain", "?") for s in sessions)
        stats["session_domain_dist"] = dict(domain_counts.most_common(10))

    # Eval instance counts
    stats["eval_counts"] = DATA.get("eval_counts", {})

    return stats


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MemArena-L Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }
  h1 { font-size: 24px; color: #fff; margin-bottom: 24px; }
  h2 { font-size: 18px; color: #6366f1; margin: 24px 0 12px 0; border-bottom: 1px solid #2a2d3a; padding-bottom: 8px; }
  h3 { font-size: 14px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; margin: 16px 0 8px 0; }

  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }
  .card { background: #1a1d28; border: 1px solid #2a2d3a; border-radius: 12px; padding: 20px; }
  .card .label { font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }
  .card .value { font-size: 32px; color: #fff; font-weight: 700; margin: 4px 0; }
  .card .sub { font-size: 13px; color: #6b7280; }

  .bar-chart { margin: 8px 0; }
  .bar-row { display: flex; align-items: center; margin: 4px 0; font-size: 13px; }
  .bar-label { width: 140px; color: #9ca3af; flex-shrink: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .bar-track { flex: 1; height: 20px; background: #1e2030; border-radius: 4px; overflow: hidden; margin: 0 8px; }
  .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
  .bar-count { width: 50px; text-align: right; color: #d1d5db; font-family: monospace; }

  .event-list { max-height: 600px; overflow-y: auto; }
  .event-item { background: #141620; border: 1px solid #1e2030; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
  .event-item .event-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
  .event-item .event-day { font-size: 12px; color: #6366f1; font-weight: 600; }
  .event-item .event-cat { font-size: 11px; padding: 2px 8px; border-radius: 4px; font-weight: 600; }
  .cat-global { background: #7c3aed22; color: #a78bfa; }
  .cat-community { background: #05966922; color: #6ee7b7; }
  .cat-dyadic { background: #d9770622; color: #fbbf24; }
  .cat-private { background: #dc262622; color: #fca5a5; }
  .event-item .event-content { font-size: 14px; color: #d1d5db; line-height: 1.5; }
  .event-item .event-meta { font-size: 12px; color: #6b7280; margin-top: 6px; }

  .persona-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 8px; max-height: 400px; overflow-y: auto; }
  .persona-chip { background: #141620; border: 1px solid #1e2030; border-radius: 8px; padding: 10px 14px; font-size: 13px; }
  .persona-chip .name { color: #fff; font-weight: 600; }
  .persona-chip .occ { color: #9ca3af; font-size: 12px; }

  .tab-bar { display: flex; gap: 4px; margin-bottom: 16px; }
  .tab { padding: 8px 20px; border-radius: 8px 8px 0 0; border: 1px solid #2a2d3a; border-bottom: none; background: #141620; color: #9ca3af; cursor: pointer; font-size: 13px; font-weight: 600; }
  .tab.active { background: #1a1d28; color: #6366f1; }
  .tab-content { display: none; }
  .tab-content.active { display: block; }

  .status-badge { display: inline-block; padding: 2px 10px; border-radius: 10px; font-size: 11px; font-weight: 600; }
  .status-badge.ok { background: #064e3b; color: #34d399; }
  .status-badge.pending { background: #78350f; color: #fbbf24; }
  .status-badge.none { background: #374151; color: #9ca3af; }
</style>
</head>
<body>

<h1>MemArena-L Dashboard</h1>

<div class="tab-bar">
  <div class="tab active" onclick="switchTab('overview')">Overview</div>
  <div class="tab" onclick="switchTab('events')">Events</div>
  <div class="tab" onclick="switchTab('personas')">Personas</div>
  <div class="tab" onclick="switchTab('schedules')">Schedules</div>
  <div class="tab" onclick="switchTab('corpus')">Corpus</div>
</div>

<div id="content"></div>

<script>
let stats = {};
let currentTab = 'overview';

async function loadStats() {
  const resp = await fetch('/api/stats');
  stats = await resp.json();
  render();
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[onclick="switchTab('${tab}')"]`).classList.add('active');
  render();
}

function barChart(data, color) {
  if (!data || Object.keys(data).length === 0) return '<p style="color:#6b7280;font-size:13px;">No data</p>';
  const max = Math.max(...Object.values(data));
  return '<div class="bar-chart">' + Object.entries(data).map(([k,v]) =>
    `<div class="bar-row">
      <span class="bar-label" title="${k}">${k}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(v/max*100)}%;background:${color};"></div></div>
      <span class="bar-count">${v}</span>
    </div>`
  ).join('') + '</div>';
}

function statusBadge(approved, pending) {
  if (approved + pending === 0) return '<span class="status-badge none">N/A</span>';
  if (pending === 0) return `<span class="status-badge ok">${approved}/${approved} approved</span>`;
  return `<span class="status-badge pending">${approved}/${approved+pending} approved</span>`;
}

function renderOverview() {
  const s = stats;
  return `
    <div class="grid">
      <div class="card">
        <div class="label">Agents</div>
        <div class="value">${s.n_agents || 0}</div>
        <div class="sub">Age: ${s.age_range || '?'} (avg ${s.age_avg || '?'}) ${statusBadge(s.personas_approved||0, s.personas_pending||0)}</div>
      </div>
      <div class="card">
        <div class="label">Events</div>
        <div class="value">${s.n_events || 0}</div>
        <div class="sub">Avg visibility: ${s.event_avg_visibility || '?'} agents</div>
      </div>
      <div class="card">
        <div class="label">Locations</div>
        <div class="value">${s.n_locations || 0}</div>
        <div class="sub">${Object.entries(s.location_type_dist||{}).map(([k,v])=>k.replace(/_/g,' ')+': '+v).join(', ')}</div>
      </div>
      <div class="card">
        <div class="label">Groups</div>
        <div class="value">${s.n_groups || 0}</div>
        <div class="sub">${Object.entries(s.group_type_dist||{}).map(([k,v])=>k+': '+v).join(', ')}</div>
      </div>
      <div class="card">
        <div class="label">Schedule Entries</div>
        <div class="value">${(s.total_schedule_entries||0).toLocaleString()}</div>
        <div class="sub">Avg ${s.avg_entries_per_agent||'?'}/agent ${statusBadge(s.schedules_approved||0, s.schedules_pending||0)}</div>
      </div>
      <div class="card">
        <div class="label">Corpus Sessions</div>
        <div class="value">${(s.n_sessions||0).toLocaleString()}</div>
        <div class="sub">${s.n_sessions ? s.total_turns.toLocaleString()+' turns, ~'+s.total_tokens_est.toLocaleString()+' tokens' : 'Not generated yet'}</div>
      </div>
      <div class="card">
        <div class="label">Tokens/Person/Day</div>
        <div class="value">${s.tokens_per_person_day ? s.tokens_per_person_day.toLocaleString() : '—'}</div>
        <div class="sub">Target: 10,000</div>
      </div>
      <div class="card">
        <div class="label">Eval Instances</div>
        <div class="value">${Object.values(s.eval_counts||{}).reduce((a,b)=>a+b, 0)}</div>
        <div class="sub">${Object.keys(s.eval_counts||{}).length} dimensions</div>
      </div>
    </div>

    <h2>Event Categories</h2>
    ${barChart(s.event_category_dist, '#6366f1')}

    <h2>Event Interest Domains</h2>
    ${barChart(s.event_domain_dist, '#059669')}

    <h2>Events Per Day</h2>
    ${barChart(Object.fromEntries(Object.entries(s.events_per_day||{}).map(([k,v])=>['Day '+(parseInt(k)+1),v])), '#f59e0b')}

    <h2>Schedule Entry Types</h2>
    ${barChart(s.schedule_type_dist, '#8b5cf6')}

    ${s.n_sessions ? `
    <h2>Session Modality</h2>
    ${barChart(s.modality_dist, '#ec4899')}

    <h2>Session Types</h2>
    ${barChart(s.session_type_dist, '#06b6d4')}
    ` : ''}

    ${Object.keys(s.eval_counts||{}).length ? `
    <h2>Eval Instances by Dimension</h2>
    ${barChart(s.eval_counts, '#f43f5e')}
    ` : ''}
  `;
}

function renderEvents() {
  const events = stats.event_list || [];
  return `
    <h2>All Events (${events.length})</h2>
    <div class="event-list">
      ${events.map(e => `
        <div class="event-item">
          <div class="event-header">
            <span class="event-day">Day ${e.day} &middot; ${e.event_type}</span>
            <span class="event-cat cat-${e.category}">${e.category}</span>
          </div>
          <div class="event-content">${e.content}</div>
          <div class="event-meta">
            Domain: ${e.interest_domain} &middot;
            Visible to: ${e.visibility_count} agent${e.visibility_count!==1?'s':''} &middot;
            Location: ${e.location_id.replace(/^loc_/,'').replace(/_/g,' ')}
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderPersonas() {
  const s = stats;
  return `
    <h2>Agents (${s.n_agents || 0})</h2>
    <div class="grid">
      <div class="card"><h3>Gender</h3>${barChart(s.gender_dist, '#6366f1')}</div>
      <div class="card"><h3>Education</h3>${barChart(s.edu_dist, '#059669')}</div>
      <div class="card"><h3>Communication Style</h3>${barChart(s.comm_dist, '#f59e0b')}</div>
    </div>
    <h3>All Agents</h3>
    <div class="persona-grid">
      ${(s.occupations||[]).map(p => `
        <div class="persona-chip">
          <div class="name">${p.name}</div>
          <div class="occ">${p.occupation}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function renderSchedules() {
  const s = stats;
  return `
    <h2>Schedules (${s.n_schedules || 0})</h2>
    <div class="grid">
      <div class="card">
        <div class="label">Total Entries</div>
        <div class="value">${(s.total_schedule_entries||0).toLocaleString()}</div>
        <div class="sub">Avg ${s.avg_entries_per_agent||'?'} per agent across 15 days</div>
      </div>
      <div class="card">
        <div class="label">Review Status</div>
        <div class="value">${s.schedules_approved||0} / ${(s.schedules_approved||0)+(s.schedules_pending||0)}</div>
        <div class="sub">approved</div>
      </div>
    </div>
    <h3>Entry Types</h3>
    ${barChart(s.schedule_type_dist, '#8b5cf6')}
  `;
}

function renderCorpus() {
  const s = stats;
  if (!s.n_sessions) return '<div class="card"><p style="color:#9ca3af;">Corpus not generated yet. Run Stage 5 to generate.</p></div>';
  return `
    <h2>Corpus Statistics</h2>
    <div class="grid">
      <div class="card">
        <div class="label">Sessions</div>
        <div class="value">${s.n_sessions.toLocaleString()}</div>
      </div>
      <div class="card">
        <div class="label">Total Turns</div>
        <div class="value">${s.total_turns.toLocaleString()}</div>
        <div class="sub">Avg ${s.avg_turns_per_session}/session, ${s.avg_words_per_turn} words/turn</div>
      </div>
      <div class="card">
        <div class="label">Estimated Tokens</div>
        <div class="value">${s.total_tokens_est.toLocaleString()}</div>
      </div>
      <div class="card">
        <div class="label">Tokens/Person/Day</div>
        <div class="value">${s.tokens_per_person_day.toLocaleString()}</div>
        <div class="sub">Target: 10,000</div>
      </div>
    </div>
    <h3>Modality</h3>
    ${barChart(s.modality_dist, '#ec4899')}
    <h3>Session Types</h3>
    ${barChart(s.session_type_dist, '#06b6d4')}
    <h3>Interest Domains</h3>
    ${barChart(s.session_domain_dist, '#10b981')}
    ${Object.keys(s.eval_counts||{}).length ? `
    <h3>Eval Instances</h3>
    ${barChart(s.eval_counts, '#f43f5e')}
    ` : ''}
  `;
}

function render() {
  const el = document.getElementById('content');
  switch (currentTab) {
    case 'overview': el.innerHTML = renderOverview(); break;
    case 'events': el.innerHTML = renderEvents(); break;
    case 'personas': el.innerHTML = renderPersonas(); break;
    case 'schedules': el.innerHTML = renderSchedules(); break;
    case 'corpus': el.innerHTML = renderCorpus(); break;
  }
}

loadStats();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[dashboard] {args[0]} {args[1]}\n")

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
        elif path == "/api/stats":
            self._json_response(compute_stats())
        else:
            self.send_error(404)


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser(description="MemArena-L Dashboard")
    parser.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")
    parser.add_argument("--port", "-p", type=int, default=10002)
    args = parser.parse_args()

    RUN_DIR = Path(args.run)
    load_data(RUN_DIR)

    n_p = len(DATA.get("personas", []))
    n_e = len(DATA.get("events", []))
    n_s = len(DATA.get("sessions", []))
    print(f"Loaded: {n_p} personas, {n_e} events, {n_s} sessions")

    server = HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"\nServing on http://0.0.0.0:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
