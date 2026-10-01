#!/usr/bin/env python3
"""Schedule Review Tool — browser-based UI for reviewing & refining agent schedules.

Usage:
    python3 review_schedules.py --run MASim/runs/l_20260408_111046 --port 10001

Then open http://<server-ip>:10001 in your browser.
"""

import argparse
import json
import re
import shutil
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

RUN_DIR: Path = Path(".")
SCHEDULES: list = []         # list of {agent_id, entries}
PERSONAS: dict = {}          # agent_id -> persona dict
STATUS: dict = {}            # agent_id -> "pending" | "approved"
LLM_ENDPOINT = "http://127.0.0.1:8000/v1"


def load_data(run_dir: Path):
    global SCHEDULES, PERSONAS, STATUS
    # Load schedules
    SCHEDULES = []
    with open(run_dir / "agent_schedules.jsonl") as f:
        for line in f:
            SCHEDULES.append(json.loads(line))
    # Load personas
    PERSONAS = {}
    with open(run_dir / "agents_personas.jsonl") as f:
        for line in f:
            row = json.loads(line)
            PERSONAS[row["agent_id"]] = row["persona"]
    # Load review status
    status_path = run_dir / "schedule_review_status.json"
    if status_path.exists():
        STATUS = json.loads(status_path.read_text())
    else:
        STATUS = {s["agent_id"]: "pending" for s in SCHEDULES}


def save_schedules():
    path = RUN_DIR / "agent_schedules.jsonl"
    backup = RUN_DIR / f"agent_schedules.jsonl.bak.{int(time.time())}"
    shutil.copy2(path, backup)
    with open(path, "w") as f:
        for row in SCHEDULES:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_status()


def save_status():
    (RUN_DIR / "schedule_review_status.json").write_text(json.dumps(STATUS, indent=2))


def refine_entry_via_llm(persona: dict, entry: dict, suggestion: str) -> str:
    """Call the LLM to refine a single schedule entry description."""
    import urllib.request

    prompt = f"""You are rewriting a schedule entry for a character in a simulation.

Character: {persona['name']}, age {persona['age']}, {persona['occupation']}
Hobbies: {', '.join(persona.get('hobbies', []))}
Daily routine: {persona.get('daily_routine_notes', 'N/A')}

Current entry:
- Type: {entry['activity_type']}
- Location: {entry['location_id']}
- Description: "{entry['description']}"

Reviewer feedback: {suggestion}

Write a revised 1-sentence description. Return ONLY the new description text, nothing else. /no_think"""

    payload = json.dumps({
        "model": "qwen3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 256,
    })

    req = urllib.request.Request(
        LLM_ENDPOINT + "/chat/completions",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())

    text = result["choices"][0]["message"]["content"].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip quotes if wrapped
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    return text


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Schedule Review Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f1117; color: #e0e0e0; }
  .header { background: #1a1d28; padding: 16px 24px; border-bottom: 1px solid #2a2d3a; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
  .header h1 { font-size: 18px; color: #fff; }
  .progress-bar { width: 300px; height: 8px; background: #2a2d3a; border-radius: 4px; overflow: hidden; }
  .progress-fill { height: 100%; background: #4ade80; transition: width 0.3s; }
  .progress-text { font-size: 13px; color: #9ca3af; margin-left: 10px; }
  .container { display: flex; height: calc(100vh - 57px); }

  .sidebar { width: 260px; background: #141620; border-right: 1px solid #2a2d3a; overflow-y: auto; flex-shrink: 0; }
  .sidebar-item { padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #1e2030; display: flex; align-items: center; gap: 8px; font-size: 13px; transition: background 0.15s; }
  .sidebar-item:hover { background: #1e2030; }
  .sidebar-item.active { background: #252840; border-left: 3px solid #6366f1; }
  .sidebar-item.approved { color: #4ade80; }
  .sidebar-item .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .sidebar-item .dot.pending { background: #f59e0b; }
  .sidebar-item .dot.approved { background: #4ade80; }

  .main { flex: 1; overflow-y: auto; padding: 32px 40px; }
  .persona-summary { background: #1a1d28; border: 1px solid #2a2d3a; border-radius: 12px; padding: 16px 20px; margin-bottom: 20px; }
  .persona-summary h2 { font-size: 22px; color: #fff; margin-bottom: 8px; }
  .persona-summary .meta { font-size: 13px; color: #9ca3af; line-height: 1.6; }
  .persona-summary .badge { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; display: inline-block; margin-left: 12px; }
  .badge.pending { background: #78350f; color: #fbbf24; }
  .badge.approved { background: #064e3b; color: #34d399; }

  .timeline { position: relative; margin-left: 20px; }
  .timeline::before { content: ''; position: absolute; left: 12px; top: 0; bottom: 0; width: 2px; background: #2a2d3a; }

  .entry { position: relative; padding: 12px 16px 12px 40px; margin-bottom: 8px; border-radius: 8px; transition: background 0.15s; }
  .entry:hover { background: #1a1d28; }
  .entry .icon { position: absolute; left: 4px; top: 14px; width: 18px; height: 18px; border-radius: 50%; border: 2px solid #2a2d3a; background: #0f1117; display: flex; align-items: center; justify-content: center; font-size: 10px; z-index: 1; }
  .entry.sleep .icon { background: #1e1b4b; border-color: #6366f1; }
  .entry.solo .icon { background: #064e3b; border-color: #4ade80; }
  .entry.transit .icon { background: #78350f; border-color: #f59e0b; }
  .entry.group_meeting .icon { background: #4a1d96; border-color: #a78bfa; }
  .entry.role_conflict .icon { background: #450a0a; border-color: #f87171; }

  .entry .time { font-size: 11px; color: #6b7280; font-family: monospace; }
  .entry .type-badge { font-size: 11px; padding: 2px 8px; border-radius: 4px; margin-left: 8px; font-weight: 600; }
  .type-badge.sleep { background: #1e1b4b; color: #a5b4fc; }
  .type-badge.solo { background: #064e3b; color: #6ee7b7; }
  .type-badge.transit { background: #78350f; color: #fcd34d; }
  .type-badge.group_meeting { background: #4a1d96; color: #c4b5fd; }
  .type-badge.role_conflict { background: #450a0a; color: #fca5a5; }

  .entry .desc { font-size: 14px; color: #d1d5db; margin-top: 4px; line-height: 1.6; }
  .entry .location { font-size: 12px; color: #6b7280; margin-top: 2px; }
  .entry .participants { font-size: 12px; color: #a78bfa; margin-top: 2px; }

  .entry .edit-btn { font-size: 11px; color: #6366f1; cursor: pointer; margin-left: 8px; opacity: 0; transition: opacity 0.15s; }
  .entry:hover .edit-btn { opacity: 1; }

  .edit-area { margin-top: 8px; display: none; }
  .edit-area textarea { width: 100%; min-height: 60px; background: #141620; color: #e0e0e0; border: 1px solid #2a2d3a; border-radius: 6px; padding: 8px; font-size: 13px; font-family: inherit; resize: vertical; }
  .edit-area textarea:focus { outline: none; border-color: #6366f1; }
  .edit-area .edit-actions { margin-top: 6px; display: flex; gap: 8px; }

  .btn { padding: 8px 20px; border-radius: 6px; border: none; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
  .btn-sm { padding: 5px 12px; font-size: 12px; }
  .btn-approve { background: #059669; color: #fff; }
  .btn-approve:hover { background: #047857; }
  .btn-refine { background: #4f46e5; color: #fff; }
  .btn-refine:hover { background: #4338ca; }
  .btn-next { background: #2a2d3a; color: #e0e0e0; }
  .btn-next:hover { background: #353849; }
  .btn-cancel { background: #374151; color: #d1d5db; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .changed { background: #1a2e1a !important; border-left: 3px solid #4ade80; }

  .actions { margin-top: 24px; display: flex; gap: 12px; }
  .nav-buttons { display: flex; gap: 8px; margin-bottom: 20px; }

  .status-msg { margin-top: 12px; padding: 10px 16px; border-radius: 8px; font-size: 13px; }
  .status-msg.info { background: #1e1b4b; color: #a5b4fc; }
  .status-msg.success { background: #064e3b; color: #6ee7b7; }
  .status-msg.error { background: #450a0a; color: #fca5a5; }
  .spinner { display: inline-block; width: 14px; height: 14px; border: 2px solid #6366f1; border-top-color: transparent; border-radius: 50%; animation: spin 0.6s linear infinite; vertical-align: middle; margin-right: 4px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>

<div class="header">
  <h1>Schedule Review</h1>
  <div style="display:flex; align-items:center;">
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <span class="progress-text" id="progressText"></span>
  </div>
</div>

<div class="container">
  <div class="sidebar" id="sidebar"></div>
  <div class="main" id="main"><p style="color:#9ca3af;">Loading...</p></div>
</div>

<script>
let schedules = [];
let personas = {};
let statuses = {};
let currentIdx = 0;
let changedEntries = {};  // idx -> Set of entry indices

async function loadAll() {
  const resp = await fetch('/api/schedules');
  const data = await resp.json();
  schedules = data.schedules;
  personas = data.personas;
  statuses = data.statuses;
  renderSidebar();
  renderSchedule(currentIdx);
  updateProgress();
}

function renderSidebar() {
  const sb = document.getElementById('sidebar');
  sb.innerHTML = schedules.map((s, i) => {
    const st = statuses[s.agent_id] || 'pending';
    const p = personas[s.agent_id] || {};
    const name = p.name || s.agent_id;
    return `<div class="sidebar-item ${i === currentIdx ? 'active' : ''} ${st}" onclick="selectAgent(${i})">
      <div class="dot ${st}"></div>
      <span>${name}</span>
    </div>`;
  }).join('');
}

function updateProgress() {
  const total = schedules.length;
  const approved = Object.values(statuses).filter(s => s === 'approved').length;
  document.getElementById('progressFill').style.width = `${(approved/total)*100}%`;
  document.getElementById('progressText').textContent = `${approved} / ${total} approved`;
}

function selectAgent(idx) {
  currentIdx = idx;
  renderSidebar();
  renderSchedule(idx);
}

function formatTime(t) {
  // Sim time: each integer = one day. Day fraction maps to clock hours.
  // e.g. t=2.375 means Day 3, hour 9:00 (0.375 * 24 = 9).
  const day = Math.floor(t) + 1;
  const dayFrac = t - Math.floor(t);
  const totalMinutes = Math.round(dayFrac * 24 * 60);
  const hour = Math.floor(totalMinutes / 60) % 24;
  const min = totalMinutes % 60;
  return `D${day} ${String(hour).padStart(2,'0')}:${String(min).padStart(2,'0')}`;
}

function renderSchedule(idx) {
  const s = schedules[idx];
  const p = personas[s.agent_id] || {};
  const st = statuses[s.agent_id] || 'pending';
  const changed = changedEntries[idx] || new Set();

  const icons = { sleep: '&#x1F319;', solo: '&#x1F464;', transit: '&#x1F697;', group_meeting: '&#x1F465;', role_conflict: '&#x26A0;' };

  let entriesHtml = s.entries.map((e, ei) => {
    const changedClass = changed.has(ei) ? 'changed' : '';
    const parts = e.participants && e.participants.length > 0
      ? `<div class="participants">with: ${e.participants.join(', ')}</div>` : '';
    return `<div class="entry ${e.activity_type} ${changedClass}" id="entry-${ei}">
      <div class="icon">${icons[e.activity_type] || '?'}</div>
      <div>
        <span class="time">${formatTime(e.start_time)} - ${formatTime(e.end_time)} (${((e.end_time - e.start_time) * 24).toFixed(1)}h)</span>
        <span class="type-badge ${e.activity_type}">${e.activity_type}</span>
        ${(e.activity_type === 'solo' || e.activity_type === 'transit') ?
          `<span class="edit-btn" onclick="toggleEdit(${idx}, ${ei})">&#9998; edit</span>` : ''}
        <div class="desc">${e.description}</div>
        <div class="location">${e.location_id.replace(/^loc_/, '').replace(/_/g, ' ')}</div>
        ${parts}
      </div>
      <div class="edit-area" id="edit-${ei}">
        <textarea id="suggestion-${ei}" placeholder="Describe what to change..."></textarea>
        <div class="edit-actions">
          <button class="btn btn-sm btn-refine" onclick="refineEntry(${idx}, ${ei})">Refine via LLM</button>
          <button class="btn btn-sm btn-cancel" onclick="toggleEdit(${idx}, ${ei})">Cancel</button>
        </div>
      </div>
    </div>`;
  }).join('');

  document.getElementById('main').innerHTML = `
    <div class="nav-buttons">
      <button class="btn btn-next" onclick="selectAgent(${Math.max(0, idx-1)})" ${idx===0?'disabled':''}>&#8592; Prev</button>
      <button class="btn btn-next" onclick="selectAgent(${Math.min(schedules.length-1, idx+1)})" ${idx===schedules.length-1?'disabled':''}>Next &#8594;</button>
      <button class="btn btn-next" onclick="goNextPending()">Next Pending</button>
    </div>

    <div class="persona-summary">
      <h2>${p.name || s.agent_id} <span class="badge ${st}">${st}</span></h2>
      <div class="meta">
        ${p.occupation || ''} &middot; Age ${p.age || '?'} &middot; ${p.work_schedule || 'N/A'} &middot; Sleep ${p.sleep_start_hour || '?'}-${p.sleep_end_hour || '?'}<br>
        Hobbies: ${(p.hobbies || []).join(', ') || 'N/A'}<br>
        Routine: ${p.daily_routine_notes || 'N/A'}
      </div>
    </div>

    <div class="timeline">${entriesHtml}</div>

    <div class="actions">
      <button class="btn btn-approve" onclick="approveSchedule(${idx})">Approve Schedule</button>
    </div>
    <div id="statusMsg"></div>
  `;
}

function toggleEdit(schedIdx, entryIdx) {
  const el = document.getElementById(`edit-${entryIdx}`);
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}

function goNextPending() {
  for (let i = currentIdx + 1; i < schedules.length; i++) {
    if (statuses[schedules[i].agent_id] !== 'approved') { selectAgent(i); return; }
  }
  for (let i = 0; i < currentIdx; i++) {
    if (statuses[schedules[i].agent_id] !== 'approved') { selectAgent(i); return; }
  }
  showStatus('All schedules approved!', 'success');
}

async function approveSchedule(idx) {
  const aid = schedules[idx].agent_id;
  const resp = await fetch(`/api/schedule/${idx}/approve`, { method: 'POST' });
  if (resp.ok) {
    statuses[aid] = 'approved';
    changedEntries[idx] = new Set();
    renderSidebar();
    renderSchedule(idx);
    updateProgress();
    showStatus('Approved! Moving to next...', 'success');
    setTimeout(() => goNextPending(), 600);
  }
}

async function refineEntry(schedIdx, entryIdx) {
  const suggestion = document.getElementById(`suggestion-${entryIdx}`).value.trim();
  if (!suggestion) { showStatus('Please enter a suggestion.', 'error'); return; }

  showStatus(`<span class="spinner"></span> Refining entry ${entryIdx+1}...`, 'info');

  try {
    const resp = await fetch(`/api/schedule/${schedIdx}/entry/${entryIdx}/refine`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ suggestion }),
    });
    if (!resp.ok) { const err = await resp.text(); throw new Error(err); }
    const data = await resp.json();
    schedules[schedIdx].entries[entryIdx].description = data.description;
    if (!changedEntries[schedIdx]) changedEntries[schedIdx] = new Set();
    changedEntries[schedIdx].add(entryIdx);
    renderSchedule(schedIdx);
    showStatus('Entry refined! Review and approve or refine again.', 'success');
  } catch (e) {
    showStatus(`Refinement failed: ${e.message}`, 'error');
  }
}

function showStatus(msg, type) {
  const el = document.getElementById('statusMsg');
  if (el) el.innerHTML = `<div class="status-msg ${type}">${msg}</div>`;
}

loadAll();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[schedule-review] {args[0]} {args[1]}\n")

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

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/":
            self._html_response(HTML_PAGE)

        elif path == "/api/schedules":
            self._json_response({
                "schedules": SCHEDULES,
                "personas": PERSONAS,
                "statuses": STATUS,
            })

        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path.endswith("/approve"):
            try:
                idx = int(path.split("/")[3])
                aid = SCHEDULES[idx]["agent_id"]
                STATUS[aid] = "approved"
                save_status()
                self._json_response({"ok": True})
            except (ValueError, IndexError):
                self._json_response({"error": "invalid index"}, 404)

        elif "/entry/" in path and path.endswith("/refine"):
            try:
                parts = path.split("/")
                sched_idx = int(parts[3])
                entry_idx = int(parts[5])
                body = json.loads(self._read_body())
                suggestion = body.get("suggestion", "")
                if not suggestion:
                    self._json_response({"error": "no suggestion"}, 400)
                    return

                agent_id = SCHEDULES[sched_idx]["agent_id"]
                persona = PERSONAS.get(agent_id, {})
                entry = SCHEDULES[sched_idx]["entries"][entry_idx]

                new_desc = refine_entry_via_llm(persona, entry, suggestion)
                SCHEDULES[sched_idx]["entries"][entry_idx]["description"] = new_desc
                save_schedules()
                self._json_response({"description": new_desc})
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif path == "/api/save":
            save_schedules()
            self._json_response({"ok": True})

        else:
            self.send_error(404)


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser(description="Schedule Review Tool")
    parser.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")
    parser.add_argument("--port", "-p", type=int, default=10001)
    args = parser.parse_args()

    RUN_DIR = Path(args.run)
    if not (RUN_DIR / "agent_schedules.jsonl").exists():
        print(f"ERROR: {RUN_DIR / 'agent_schedules.jsonl'} not found", file=sys.stderr)
        sys.exit(1)

    load_data(RUN_DIR)
    print(f"Loaded {len(SCHEDULES)} schedules from {RUN_DIR}")
    print(f"Status: {sum(1 for s in STATUS.values() if s == 'approved')} approved, "
          f"{sum(1 for s in STATUS.values() if s == 'pending')} pending")

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
