#!/usr/bin/env python3
"""Persona Review Tool — browser-based UI for reviewing & refining generated personas.

Usage:
    python3 review_personas.py --run MASim/runs/l_20260408_111046 --port 8080

Then open http://<server-ip>:8080 in your browser.
"""

import argparse
import json
import shutil
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

RUN_DIR: Path = Path(".")
PERSONAS: list = []          # list of full rows (agent_id, persona, social_neighbors, ...)
STATUS: dict = {}            # agent_id -> "pending" | "approved"
LLM_ENDPOINT = "http://127.0.0.1:8000/v1"

def load_personas(run_dir: Path):
    global PERSONAS, STATUS
    path = run_dir / "agents_personas.jsonl"
    PERSONAS = []
    with open(path) as f:
        for line in f:
            PERSONAS.append(json.loads(line))
    # Load review status if exists
    status_path = run_dir / "review_status.json"
    if status_path.exists():
        STATUS = json.loads(status_path.read_text())
    else:
        STATUS = {p["agent_id"]: "pending" for p in PERSONAS}

def save_personas():
    path = RUN_DIR / "agents_personas.jsonl"
    # Backup first
    backup = RUN_DIR / f"agents_personas.jsonl.bak.{int(time.time())}"
    shutil.copy2(path, backup)
    with open(path, "w") as f:
        for row in PERSONAS:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Save status
    (RUN_DIR / "review_status.json").write_text(json.dumps(STATUS, indent=2))

def save_status():
    (RUN_DIR / "review_status.json").write_text(json.dumps(STATUS, indent=2))

def _extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output, handling think tags and markdown fences."""
    import re
    # Strip Qwen3 <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    # Find the JSON object (first { to last })
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM response: {text[:200]}")
    return json.loads(text[start:end+1])


def refine_persona_via_llm(persona: dict, suggestion: str) -> dict:
    """Call the LLM to refine a persona based on user suggestion."""
    import urllib.request

    prompt = f"""You are editing a character profile for a simulation. Here is the current profile:

{json.dumps(persona, indent=2, ensure_ascii=False)}

The reviewer's feedback:
{suggestion}

Revise the profile based on the feedback. Keep the exact same JSON schema/keys. Only change what the feedback asks for. Return ONLY the valid JSON object, no markdown fences, no explanation. /no_think"""

    payload = json.dumps({
        "model": "qwen3",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4096,
    })

    req = urllib.request.Request(
        LLM_ENDPOINT + "/chat/completions",
        data=payload.encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())

    text = result["choices"][0]["message"]["content"].strip()
    return _extract_json(text)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Persona Review Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f1117; color: #e0e0e0; }
  .header { background: #1a1d28; padding: 16px 24px; border-bottom: 1px solid #2a2d3a; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 100; }
  .header h1 { font-size: 18px; color: #fff; }
  .progress-bar { width: 300px; height: 8px; background: #2a2d3a; border-radius: 4px; overflow: hidden; }
  .progress-fill { height: 100%; background: #4ade80; transition: width 0.3s; }
  .progress-text { font-size: 13px; color: #9ca3af; margin-left: 10px; }
  .container { display: flex; height: calc(100vh - 57px); }

  /* Sidebar */
  .sidebar { width: 260px; background: #141620; border-right: 1px solid #2a2d3a; overflow-y: auto; flex-shrink: 0; }
  .sidebar-item { padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #1e2030; display: flex; align-items: center; gap: 8px; font-size: 13px; transition: background 0.15s; }
  .sidebar-item:hover { background: #1e2030; }
  .sidebar-item.active { background: #252840; border-left: 3px solid #6366f1; }
  .sidebar-item.approved { color: #4ade80; }
  .sidebar-item .dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .sidebar-item .dot.pending { background: #f59e0b; }
  .sidebar-item .dot.approved { background: #4ade80; }

  /* Main content */
  .main { flex: 1; overflow-y: auto; padding: 32px 40px; }
  .persona-header { display: flex; align-items: center; gap: 16px; margin-bottom: 24px; }
  .persona-header h2 { font-size: 28px; color: #fff; }
  .persona-header .badge { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }
  .badge.pending { background: #78350f; color: #fbbf24; }
  .badge.approved { background: #064e3b; color: #34d399; }
  .meta-row { display: flex; gap: 24px; margin-bottom: 20px; flex-wrap: wrap; }
  .meta-chip { background: #1e2030; padding: 6px 14px; border-radius: 8px; font-size: 13px; }
  .meta-chip span { color: #9ca3af; }

  .section { margin-bottom: 20px; }
  .section h3 { font-size: 14px; color: #6366f1; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  .section p, .section li { font-size: 14px; line-height: 1.7; color: #d1d5db; }
  .section ul { padding-left: 20px; }
  .section .kvtable { width: 100%; border-collapse: collapse; }
  .section .kvtable td { padding: 4px 12px 4px 0; font-size: 13px; vertical-align: top; }
  .section .kvtable td:first-child { color: #9ca3af; width: 160px; white-space: nowrap; }

  .card { background: #1a1d28; border: 1px solid #2a2d3a; border-radius: 12px; padding: 20px; margin-bottom: 16px; }
  .changed { background: #1a2e1a; border-left: 3px solid #4ade80; padding-left: 8px; }
  .changed-chip { background: #1a2e1a !important; outline: 2px solid #4ade80; }
  .diff-banner { background: #1a2e1a; border: 1px solid #4ade80; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; font-size: 13px; color: #6ee7b7; }
  .diff-banner strong { color: #4ade80; }

  /* Actions */
  .actions { margin-top: 24px; display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
  .btn { padding: 10px 24px; border-radius: 8px; border: none; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
  .btn-approve { background: #059669; color: #fff; }
  .btn-approve:hover { background: #047857; }
  .btn-refine { background: #4f46e5; color: #fff; }
  .btn-refine:hover { background: #4338ca; }
  .btn-next { background: #2a2d3a; color: #e0e0e0; }
  .btn-next:hover { background: #353849; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .suggestion-box { width: 100%; margin-top: 12px; }
  .suggestion-box textarea { width: 100%; min-height: 100px; background: #141620; color: #e0e0e0; border: 1px solid #2a2d3a; border-radius: 8px; padding: 12px; font-size: 14px; font-family: inherit; resize: vertical; }
  .suggestion-box textarea:focus { outline: none; border-color: #6366f1; }

  .status-msg { margin-top: 12px; padding: 10px 16px; border-radius: 8px; font-size: 13px; }
  .status-msg.info { background: #1e1b4b; color: #a5b4fc; }
  .status-msg.success { background: #064e3b; color: #6ee7b7; }
  .status-msg.error { background: #450a0a; color: #fca5a5; }

  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #6366f1; border-top-color: transparent; border-radius: 50%; animation: spin 0.6s linear infinite; vertical-align: middle; margin-right: 6px; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .nav-buttons { display: flex; gap: 8px; margin-bottom: 20px; }
</style>
</head>
<body>

<div class="header">
  <h1>Persona Review</h1>
  <div style="display:flex; align-items:center;">
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <span class="progress-text" id="progressText"></span>
  </div>
</div>

<div class="container">
  <div class="sidebar" id="sidebar"></div>
  <div class="main" id="main">
    <p style="color:#9ca3af;">Loading personas...</p>
  </div>
</div>

<script>
let personas = [];
let statuses = {};
let currentIdx = 0;
let changedFields = {};  // idx -> Set of changed field names

async function loadAll() {
  const resp = await fetch('/api/personas');
  const data = await resp.json();
  personas = data.personas;
  statuses = data.statuses;
  renderSidebar();
  renderPersona(currentIdx);
  updateProgress();
}

function renderSidebar() {
  const sb = document.getElementById('sidebar');
  sb.innerHTML = personas.map((p, i) => {
    const st = statuses[p.agent_id] || 'pending';
    return `<div class="sidebar-item ${i === currentIdx ? 'active' : ''} ${st}" onclick="selectPersona(${i})">
      <div class="dot ${st}"></div>
      <span>${p.persona.name}</span>
    </div>`;
  }).join('');
}

function updateProgress() {
  const total = personas.length;
  const approved = Object.values(statuses).filter(s => s === 'approved').length;
  document.getElementById('progressFill').style.width = `${(approved/total)*100}%`;
  document.getElementById('progressText').textContent = `${approved} / ${total} approved`;
}

function selectPersona(idx) {
  currentIdx = idx;
  renderSidebar();
  renderPersona(idx);
}

function renderPersona(idx) {
  const row = personas[idx];
  const p = row.persona;
  const st = statuses[row.agent_id] || 'pending';

  const demoHtml = p.demographics ? Object.entries(p.demographics).map(
    ([k,v]) => `<tr><td>${k.replace(/_/g,' ')}</td><td>${v}</td></tr>`
  ).join('') : '';

  const cf = changedFields[idx] || new Set();
  const hl = (field) => cf.has(field) ? 'changed' : '';
  const hlChip = (field) => cf.has(field) ? 'meta-chip changed-chip' : 'meta-chip';

  const listSection = (title, items, field) => {
    if (!items || items.length === 0) return '';
    return `<div class="section ${hl(field)}"><h3>${title}</h3><ul>${items.map(i => `<li>${i}</li>`).join('')}</ul></div>`;
  };

  const textSection = (title, text, field) => {
    if (!text) return '';
    return `<div class="section ${hl(field)}"><h3>${title}</h3><p>${text}</p></div>`;
  };

  const relHtml = p.relationships ? Object.entries(p.relationships).map(
    ([k,v]) => `<tr><td>${k}</td><td>${typeof v === 'object' ? JSON.stringify(v) : v}</td></tr>`
  ).join('') : '';

  document.getElementById('main').innerHTML = `
    <div class="nav-buttons">
      <button class="btn btn-next" onclick="selectPersona(${Math.max(0, idx-1)})" ${idx===0?'disabled':''}>&#8592; Prev</button>
      <button class="btn btn-next" onclick="selectPersona(${Math.min(personas.length-1, idx+1)})" ${idx===personas.length-1?'disabled':''}>Next &#8594;</button>
      <button class="btn btn-next" onclick="goNextPending()">Next Pending</button>
    </div>

    <div class="persona-header">
      <h2>${p.name}</h2>
      <span class="badge ${st}">${st}</span>
    </div>

    ${cf.size > 0 ? `<div class="diff-banner"><strong>${cf.size} field(s) changed:</strong> ${[...cf].join(', ')}</div>` : ''}

    <div class="meta-row">
      <div class="${hlChip('age')}">Age: <strong>${p.age}</strong></div>
      <div class="${hlChip('occupation')}">${p.occupation}</div>
      <div class="${hlChip('education_level')}">Education: ${p.education_level || 'N/A'}</div>
      <div class="${hlChip('communication_style')}">Style: ${p.communication_style || 'N/A'}</div>
      <div class="${hlChip('work_schedule')}">Schedule: ${p.work_schedule || 'N/A'}</div>
      <div class="${hlChip('sleep_start_hour')}">Sleep: ${p.sleep_start_hour}-${p.sleep_end_hour}</div>
    </div>

    <div class="card">
      ${textSection('Backstory', p.backstory, 'backstory')}
      ${textSection('Speaking Style', p.speaking_style, 'speaking_style')}
      ${textSection('Daily Routine', p.daily_routine_notes, 'daily_routine_notes')}
    </div>

    <div class="card">
      ${listSection('Personality Traits', p.personality_traits, 'personality_traits')}
      ${listSection('Hobbies', p.hobbies, 'hobbies')}
      ${listSection('Values', p.values, 'values')}
      ${listSection('Expertise', p.expertise, 'expertise')}
      ${listSection('Current Concerns', p.current_concerns, 'current_concerns')}
    </div>

    ${demoHtml ? `<div class="card"><div class="section"><h3>Demographics</h3><table class="kvtable">${demoHtml}</table></div></div>` : ''}
    ${relHtml ? `<div class="card"><div class="section"><h3>Relationships</h3><table class="kvtable">${relHtml}</table></div></div>` : ''}

    <div class="actions">
      <button class="btn btn-approve" id="btnApprove" onclick="approvePersona(${idx})">Approve</button>
      <button class="btn btn-refine" onclick="toggleSuggestion()">Suggest Refinement</button>
    </div>

    <div id="suggestionArea" style="display:none;">
      <div class="suggestion-box">
        <textarea id="suggestionText" placeholder="Describe what to change... e.g. 'Make backstory more specific about their childhood in Ohio' or 'Add a hobby related to cooking'"></textarea>
      </div>
      <div style="margin-top:8px;">
        <button class="btn btn-refine" id="btnRefine" onclick="refinePersona(${idx})">Send to LLM for Refinement</button>
      </div>
    </div>

    <div id="statusMsg"></div>
  `;
}

function toggleSuggestion() {
  const area = document.getElementById('suggestionArea');
  area.style.display = area.style.display === 'none' ? 'block' : 'none';
}

function goNextPending() {
  for (let i = currentIdx + 1; i < personas.length; i++) {
    if (statuses[personas[i].agent_id] !== 'approved') { selectPersona(i); return; }
  }
  for (let i = 0; i < currentIdx; i++) {
    if (statuses[personas[i].agent_id] !== 'approved') { selectPersona(i); return; }
  }
  showStatus('All personas approved!', 'success');
}

async function approvePersona(idx) {
  const aid = personas[idx].agent_id;
  const resp = await fetch(`/api/persona/${idx}/approve`, { method: 'POST' });
  if (resp.ok) {
    statuses[aid] = 'approved';
    renderSidebar();
    renderPersona(idx);
    updateProgress();
    showStatus('Approved! Moving to next...', 'success');
    setTimeout(() => goNextPending(), 600);
  }
}

async function refinePersona(idx) {
  const suggestion = document.getElementById('suggestionText').value.trim();
  if (!suggestion) { showStatus('Please enter a suggestion first.', 'error'); return; }

  const btn = document.getElementById('btnRefine');
  btn.disabled = true;
  showStatus('<span class="spinner"></span> Refining via LLM... this may take 30-60s', 'info');

  try {
    const oldPersona = JSON.parse(JSON.stringify(personas[idx].persona));
    const resp = await fetch(`/api/persona/${idx}/refine`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ suggestion }),
    });
    if (!resp.ok) { const err = await resp.text(); throw new Error(err); }
    const data = await resp.json();
    personas[idx].persona = data.persona;

    // Compute changed fields
    const changed = new Set();
    const newP = data.persona;
    for (const key of Object.keys(newP)) {
      const o = JSON.stringify(oldPersona[key]);
      const n = JSON.stringify(newP[key]);
      if (o !== n) changed.add(key);
    }
    changedFields[idx] = changed;

    renderPersona(idx);
    showStatus(`Persona refined! ${changed.size} field(s) changed: ${[...changed].join(', ')}. Review and approve or refine again.`, 'success');
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
        # Quieter logging
        sys.stderr.write(f"[review] {args[0]} {args[1]}\n")

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

        elif path == "/api/personas":
            summary = []
            for row in PERSONAS:
                summary.append({
                    "agent_id": row["agent_id"],
                    "persona": row["persona"],
                })
            self._json_response({"personas": summary, "statuses": STATUS})

        elif path.startswith("/api/persona/"):
            try:
                idx = int(path.split("/")[3])
                self._json_response(PERSONAS[idx])
            except (ValueError, IndexError):
                self._json_response({"error": "invalid index"}, 404)

        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path.endswith("/approve"):
            try:
                idx = int(path.split("/")[3])
                aid = PERSONAS[idx]["agent_id"]
                STATUS[aid] = "approved"
                save_status()
                self._json_response({"ok": True})
            except (ValueError, IndexError):
                self._json_response({"error": "invalid index"}, 404)

        elif path.endswith("/refine"):
            try:
                idx = int(path.split("/")[3])
                body = json.loads(self._read_body())
                suggestion = body.get("suggestion", "")
                if not suggestion:
                    self._json_response({"error": "no suggestion"}, 400)
                    return

                old_persona = PERSONAS[idx]["persona"]
                new_persona = refine_persona_via_llm(old_persona, suggestion)
                PERSONAS[idx]["persona"] = new_persona
                save_personas()
                self._json_response({"persona": new_persona})
            except Exception as e:
                self._json_response({"error": str(e)}, 500)

        elif path == "/api/save":
            save_personas()
            self._json_response({"ok": True})

        else:
            self.send_error(404)


def main():
    global RUN_DIR
    parser = argparse.ArgumentParser(description="Persona Review Tool")
    parser.add_argument("--run", "-r", required=True, help="Path to pipeline run directory")
    parser.add_argument("--port", "-p", type=int, default=8080)
    args = parser.parse_args()

    RUN_DIR = Path(args.run)
    if not (RUN_DIR / "agents_personas.jsonl").exists():
        print(f"ERROR: {RUN_DIR / 'agents_personas.jsonl'} not found", file=sys.stderr)
        sys.exit(1)

    load_personas(RUN_DIR)
    print(f"Loaded {len(PERSONAS)} personas from {RUN_DIR}")
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
