#!/usr/bin/env python3
'''Trial-status dashboard. Reads trials_state.json, parses logs for live
progress, computes rate-based ETA. Pages: /, /gpu, /logs.'''
import json, time, os, re, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from datetime import datetime
from judge_status import judge_status, judge_accuracy_table

STATE_FILE = os.path.expanduser('~/MemArena/trials_state.json')
LOG_DIR = '/tmp'
TOTAL_INSTANCES = 1831
PORT = 8765

def load_state():
    return json.load(open(STATE_FILE))

def fmt_min(m):
    if m is None or m < 0: return '—'
    if m < 60: return f'{m:.0f}m'
    h = int(m // 60); mm = int(m % 60)
    return f'{h}h{mm:02d}m'

PROGRESS_PATTERNS = [
    re.compile(r"\[timing\]\s+(\d+)/(\d+)\s+acc="),
    re.compile(r"\[eval_search\]\s*\|[#-]+\|\s*(\d+)/(\d+)"),
    re.compile(r"\[eval_answer\]\s*\|[#-]+\|\s*(\d+)/(\d+)"),
    re.compile(r"\[bulk\]\s+(\d+)/(\d+)\s+acc="),
    re.compile(r'(?:Answered|Processed|Done)[: ]+(\d+)\s*/\s*(\d+)'),
    re.compile(r'\[(\d+)/(\d+)\]'),
    re.compile(r'(\d+)/(\d+)\s+(?:\[|complete|done)'),
    re.compile(r'progress[: ]+(\d+)\s*/\s*(\d+)', re.I),
    re.compile(r'(\d+)\s*it[/\\]s'),  # tqdm
]

def parse_log_progress(log_path):
    '''Return (done, total) or (None, None) if no progress info.'''
    if not log_path or not os.path.exists(log_path):
        return None, None
    try:
        # Read last 50KB only
        size = os.path.getsize(log_path)
        with open(log_path,'rb') as f:
            if size > 50000: f.seek(-50000, 2)
            tail = f.read().decode('utf-8','ignore')
    except Exception:
        return None, None
    last_done = None
    for pat in PROGRESS_PATTERNS[:4]:
        for m in pat.finditer(tail):
            last_done = (int(m.group(1)), int(m.group(2)))
    return last_done if last_done else (None, None)

def get_accuracy(rp):
    if not rp: return None, 'n/a'
    if not os.path.exists(rp): return None, 'no file'
    try:
        d=json.load(open(rp))
        acc=d.get('summary',{}).get('accuracy')
        if acc is None:
            for k in ('accuracy','overall_accuracy','acc'):
                if k in d: acc=d[k]; break
        if acc is None: return None, 'no acc'
        return acc, None
    except Exception as e:
        return None, f'err:{type(e).__name__}'


def nvidia_smi():
    try:
        return subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT, timeout=5).decode()
    except Exception as e:
        return f'nvidia-smi error: {e}'

def docker_logs():
    try:
        names = subprocess.check_output(['docker','ps','--filter','name=sglang','--format','{{.Names}}'], timeout=5).decode().split()
    except Exception as e:
        return 'docker ps error: ' + str(e)
    if not names: return '(no sglang containers running)'
    out=[]
    for n in names:
        try:
            log = subprocess.check_output(['docker','logs','--tail','15',n], stderr=subprocess.STDOUT, timeout=5).decode()
        except Exception as e:
            log = 'error: ' + str(e)
        out.append('=== ' + n + ' ===' + chr(10) + log)
    return chr(10).join(out)

def status_badge(s):
    colors = {'done':'#2a8a4f','running':'#e0a000','pending':'#888','failed':'#c0392b','waiting':'#5b8def'}
    return f'<span style="background:{colors.get(s,"#888")};color:#fff;padding:2px 8px;border-radius:10px;font-size:12px;">{s.upper()}</span>'

def trial_row(t, now):
    s = t['status']
    log_path = t.get('log_path') or f'{LOG_DIR}/{t["id"]}.log'
    done, total = parse_log_progress(log_path) if s == 'running' else (None, None)
    if total is None: total = t.get('total_instances') or TOTAL_INSTANCES
    if t.get('total_instances'): total = t['total_instances']

    # Progress + ETA
    if s == 'running':
        elapsed = (now - t.get('started_at', now)) / 60
        if done and done > 0 and elapsed > 0.1:
            rate = done / elapsed   # instances per minute
            remaining = (total - done) / rate if rate > 0 else None
            prog = f'{done}/{total} &nbsp;({elapsed:.1f}m, {rate:.1f}/m)'
            eta_cell = f'~{fmt_min(remaining)}'
        else:
            prog = f'0/{total} &nbsp;({elapsed:.1f}m, warming up)'
            eta_cell = '—'
    elif s == 'done':
        dur = (t.get('finished_at',0) - t.get('started_at',0)) / 60
        prog = f'{total}/{total}'
        eta_cell = f'took {fmt_min(dur)}'
    elif s == 'failed':
        prog = 'failed'
        eta_cell = '—'
    else:
        prog = f'0/{total}'
        eta_cell = f'~{fmt_min(t.get("eta_min"))}'

    acc, err = get_accuracy(t.get('result_path'))
    if acc is not None:
        acc_cell = f'<b>{acc*100:.1f}%</b>' if acc<=1 else f'<b>{acc:.4f}</b>'
    elif s == 'done':
        acc_cell = f'<span style="color:#c0392b">incomplete ({err})</span>'
    elif s == 'running':
        acc_cell = '<span style="color:#888">…</span>'
    else:
        acc_cell = '—'

    notes = t.get('notes','')
    return f'<tr><td>{t["id"]}</td><td>{t["backend"]}</td><td>{t["model"]}</td><td>{status_badge(s)}</td><td>{prog}</td><td>{eta_cell}</td><td>{acc_cell}</td><td style="color:#888;font-size:12px">{notes}</td></tr>'

def render_trials():
    st = load_state()
    trials = st.get('trials', [])
    now = time.time()
    rows = ''.join(trial_row(t, now) for t in trials)
    done = sum(1 for t in trials if t['status']=='done')
    running = sum(1 for t in trials if t['status']=='running')
    failed = sum(1 for t in trials if t['status']=='failed')
    total = len(trials)

    # Rate-based grand ETA: max of running ETAs (parallel) + sum of pending ETAs (sequential? or parallel?)
    # Simpler: sum eta_min of pending + max running ETA
    pending_eta = sum((t.get('eta_min') or 0) for t in trials if t['status']=='pending')
    last_update = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    nav = '<p><a href="/">Trials</a> | <a href="/gpu">GPU</a> | <a href="/logs">Docker logs</a></p>'

    return f'''<!DOCTYPE html><html><head><meta charset="UTF-8">
<meta http-equiv="refresh" content="10">
<title>L-Scale Trial Dashboard</title>
<style>
body {{font-family:-apple-system,system-ui,sans-serif;max-width:1200px;margin:1em auto;padding:0 1em;color:#222;}}
h1 {{border-bottom:2px solid #444;padding-bottom:.3em;}}
table {{border-collapse:collapse;width:100%;margin:.4em 0 1em;}}
th,td {{border:1px solid #ccc;padding:5px 9px;text-align:left;font-size:13px;}}
th {{background:#f0f4fa;}}
tr:nth-child(even) td {{background:#fafafa;}}
.summary {{background:#fffbe6;padding:.6em 1em;border-left:4px solid #e0c000;margin:1em 0;}}
.meta {{color:#888;font-size:12px;}}
</style></head><body>
<h1>MemArena-L Trial Dashboard</h1>
{nav}
<p class="meta">Run dir: <code>MASim/runs/l_20260408_111046</code> &nbsp;|&nbsp; Total instances per trial: {TOTAL_INSTANCES} &nbsp;|&nbsp; Last update: {last_update}</p>
<div class="summary"><b>{done}/{total}</b> done &nbsp;|&nbsp; <b>{running}</b> running &nbsp;|&nbsp; <b>{failed}</b> failed &nbsp;|&nbsp; pending ETA sum: ~{fmt_min(pending_eta)}</div>
<table>
<tr><th>ID</th><th>Backend</th><th>Model</th><th>Status</th><th>Progress</th><th>ETA</th><th>Accuracy</th><th>Notes</th></tr>
{rows}
</table>
<p class="meta">ETA for running trials is rate-based (instances/min from log). Pending ETAs are static estimates from <code>trials_state.json</code>.</p>
</body></html>'''

def render_page(title, body):
    nav = '<p><a href="/">Trials</a> | <a href="/gpu">GPU</a> | <a href="/logs">Docker logs</a></p>'
    return f'<!DOCTYPE html><html><head><meta charset="UTF-8"><meta http-equiv="refresh" content="10"><title>{title}</title><style>body{{font-family:-apple-system,system-ui,sans-serif;max-width:1200px;margin:1em auto;padding:0 1em;}}h1{{border-bottom:2px solid #444;padding-bottom:.3em;}}pre{{padding:10px;font-size:11px;overflow-x:auto;}}</style></head><body><h1>{title}</h1>{nav}{body}</body></html>'

class H(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path.startswith('/gpu'):
                body = '<pre style="background:#111;color:#0f0;">' + nvidia_smi() + '</pre>'
                html = render_page('GPU (nvidia-smi)', body)
            elif self.path.startswith('/judge'):
                body = '<pre style="background:#1a1a2e;color:#e0e0ff;padding:12px;font-size:13px;">' + judge_status() + "</pre><h2>Accuracy (L-scale)</h2><pre style=\"background:#0a0a1a;color:#90ff90;padding:12px;font-size:13px;\">" + judge_accuracy_table() + "</pre>"
                html = render_page('LLM-as-a-Judge (GPT-4o-mini)', body)
            elif self.path.startswith('/logs') or self.path.startswith('/docker'):
                body = '<pre style="background:#111;color:#0ff;">' + docker_logs() + '</pre>'
                html = render_page('Docker logs (sglang containers)', body)
            else:
                html = render_trials()
            self.send_response(200)
            self.send_header('Content-Type','text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode())
        except Exception as e:
            import traceback
            self.send_response(500); self.end_headers()
            self.wfile.write(traceback.format_exc().encode())
    def log_message(self, *a, **k): pass

if __name__ == '__main__':
    print(f'Serving on http://0.0.0.0:{PORT}')
    HTTPServer(('0.0.0.0', PORT), H).serve_forever()
