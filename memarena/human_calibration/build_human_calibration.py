#!/usr/bin/env python3
"""Build a self-contained HTML for 500-instance human judge calibration.

Samples instances stratified by (dimension x judge_correct) across cells, then
embeds the sample pool in a static HTML file. Open the HTML, click right/wrong
for each item; the page autosaves to localStorage and lets the user download a
labels.json at any time for downstream kappa analysis.

Usage
-----
    python3 scripts/build_human_calibration.py \
        --run-dir MASim/runs/l_20260408_111046 \
        --judge 4omini \
        --n 500 \
        --seed 20260419 \
        --out docs/human_calibration/index.html
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path


CELL_RE = re.compile(r"evaluation_results_(?P<cell>.+)_(?P<judge>4omini|qwen3)\.json$")


def discover_eval_files(root: Path, judge: str):
    patterns = [
        root / "eval_results" / "**" / f"evaluation_results_*_{judge}.json",
        root / "eval_results" / "memory_cache" / "memory_cache"
            / f"evaluation_results_*_{judge}.json",
    ]
    seen = set()
    files = []
    for pat in patterns:
        for p in glob.glob(str(pat), recursive=True):
            if p in seen:
                continue
            seen.add(p)
            files.append(Path(p))
    return sorted(files)


def load_rows(files):
    by_cell = {}
    for f in files:
        m = CELL_RE.search(f.name)
        if not m:
            continue
        cell = m.group("cell")
        try:
            data = json.loads(f.read_text())
        except Exception as exc:
            print(f"  skip {f}: {exc}", file=sys.stderr)
            continue
        rows = data.get("details", [])
        if not rows:
            continue
        by_cell[cell] = rows
    return by_cell


JUDGE_SCORING_METHOD = "judge"


def filter_pool(by_cell, *, judge_only: bool):
    """Return {(cell, instance_id) -> (cell, row)} pool after filtering.

    When judge_only is True, keep only rows actually scored by the LLM judge
    (scoring_method == 'judge'). Paper §5 / Appendix L scope human calibration
    to the judge-scored subset (D2–D4 and the answer-required half of D5);
    D1 / D6 / D5-abstain / pre_scored:LLM_ERROR are deterministic and must be
    excluded from kappa.
    """
    pool = {}
    for cell, rows in by_cell.items():
        for r in rows:
            if not r.get("instance_id"):
                continue
            if judge_only and r.get("scoring_method") != JUDGE_SCORING_METHOD:
                continue
            # Skip rows without a usable judge verdict
            jc = r.get("judge_correct")
            if jc is None:
                jc = r.get("correct")
            if jc is None:
                continue
            pool[(cell, r["instance_id"])] = (cell, r)
    return pool


def build_full_pool(by_cell):
    """Unfiltered pool keyed by (cell, instance_id) — used to recover pinned rows."""
    pool = {}
    for cell, rows in by_cell.items():
        for r in rows:
            iid = r.get("instance_id")
            if not iid:
                continue
            pool[(cell, iid)] = (cell, r)
    return pool


def stratified_sample(pool, n, rng, *, pinned_ids=None):
    """Stratified sample of size `n` from `pool` with optional pinned IDs.

    pinned_ids: set of instance_id strings that MUST appear in the output
                (already-judged rows to preserve). Pinned rows don't count
                toward bucket quotas but do count toward `n`.
    """
    pinned_ids = set(pinned_ids or [])

    # Dedupe globally by instance_id: the label file is keyed by id only, so
    # picking the same id from multiple cells would cause label collisions.
    # For pinned ids, keep the first cell seen; for unpinned, pick one cell
    # at random (rng.choice) to keep the sample representative of cell mix.
    by_id = defaultdict(list)  # iid -> [(cell, r), ...]
    for key, (cell, r) in sorted(pool.items()):
        by_id[r.get("instance_id")].append((cell, r))

    pinned_rows = []
    unpinned = {}
    for iid, rows in by_id.items():
        if iid in pinned_ids:
            pinned_rows.append(rows[0])  # deterministic: sorted by (cell, id)
        else:
            cell, r = rng.choice(rows)
            unpinned[(cell, iid)] = (cell, r)

    # Group unpinned rows by (dimension, judge_correct).
    buckets = defaultdict(list)
    for cell, r in unpinned.values():
        dim = r.get("dimension") or "unknown"
        jc = r.get("judge_correct")
        if jc is None:
            jc = r.get("correct")
        buckets[(dim, bool(jc))].append((cell, r))

    keys = sorted(buckets.keys())
    if not keys and not pinned_rows:
        raise SystemExit("empty pool after filtering")

    need = max(0, n - len(pinned_rows))
    per = max(1, need // max(1, len(keys))) if keys else 0

    sample = list(pinned_rows)
    leftovers = []

    for key in keys:
        rng.shuffle(buckets[key])
        take = min(per, len(buckets[key]))
        sample.extend(buckets[key][:take])
        if len(buckets[key]) > take:
            leftovers.extend(buckets[key][take:])

    # Fill to n with leftovers — covers both the rounding slack and any
    # under-full buckets (some (dim, jc) cells are small after global id dedup).
    rng.shuffle(leftovers)
    shortfall = n - len(sample)
    if shortfall > 0:
        sample.extend(leftovers[:shortfall])

    rng.shuffle(sample)
    return sample[:n]


def build_payload(sample):
    items = []
    for cell, r in sample:
        jc = r.get("judge_correct")
        if jc is None:
            jc = r.get("correct")
        items.append({
            "id": r.get("instance_id"),
            "cell": cell,
            "dim": r.get("dimension"),
            "query": r.get("query") or "",
            "gold": r.get("gold") or "",
            "prediction": r.get("prediction") or "",
            "judge_correct": bool(jc),
            "judge_reason": r.get("judge_reason") or r.get("reason") or "",
        })
    return items


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>Human Judge Calibration — MemArena</title>
<style>
  :root {
    --bg:#0e1116; --panel:#161b22; --border:#30363d; --text:#e6edf3;
    --muted:#8b949e; --accent:#1f6feb; --good:#238636; --bad:#da3633;
    --skip:#6e7681;
  }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
         background:var(--bg); color:var(--text); }
  header { padding:12px 20px; border-bottom:1px solid var(--border);
           display:flex; justify-content:space-between; align-items:center;
           background:var(--panel); position:sticky; top:0; z-index:10; }
  header .left { display:flex; gap:16px; align-items:center; }
  header .counter { font-weight:600; font-size:16px; }
  header .progress { width:240px; height:8px; background:var(--border);
                     border-radius:4px; overflow:hidden; }
  header .progress > div { height:100%; background:var(--accent); width:0; transition:width .2s; }
  header .kappa { font-family:ui-monospace,monospace; color:var(--muted); font-size:13px; }
  header button { background:var(--panel); color:var(--text); border:1px solid var(--border);
                  padding:6px 12px; border-radius:6px; cursor:pointer; font-size:13px; }
  header button:hover { border-color:var(--accent); }

  main { max-width:920px; margin:20px auto; padding:0 20px; }
  .card { background:var(--panel); border:1px solid var(--border); border-radius:10px;
          padding:24px; margin-bottom:16px; }
  .meta { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:16px; color:var(--muted); font-size:13px; }
  .badge { background:#21262d; border:1px solid var(--border); border-radius:4px; padding:2px 8px;
           font-family:ui-monospace,monospace; font-size:12px; color:var(--text); }
  .block { margin-bottom:16px; }
  .block .label { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.08em;
                  margin-bottom:6px; font-weight:600; }
  .block .body { font-size:15px; line-height:1.55; white-space:pre-wrap; word-wrap:break-word; }
  .query  .body { color:#f0f6fc; }
  .gold   .body { color:#7ee787; }
  .pred   .body { color:#ffa657; }

  .judge { border-top:1px dashed var(--border); padding-top:12px; margin-top:12px;
           color:var(--muted); font-size:13px; display:none; }
  .judge.shown { display:block; }
  .judge .verdict { font-family:ui-monospace,monospace; color:#f0f6fc; }
  .judge .verdict.t { color:var(--good); }
  .judge .verdict.f { color:var(--bad); }

  .controls { display:flex; gap:12px; justify-content:center; margin:20px 0 12px; flex-wrap:wrap; }
  .controls button { font-size:15px; padding:10px 24px; border-radius:8px; border:1px solid var(--border);
                     color:#fff; cursor:pointer; font-weight:600; min-width:120px; }
  .btn-good { background:var(--good); }
  .btn-bad  { background:var(--bad);  }
  .btn-skip { background:var(--skip); }
  .controls button:hover { filter:brightness(1.1); }

  .navrow { display:flex; justify-content:space-between; align-items:center; margin-top:12px; color:var(--muted); font-size:13px; }
  .navrow button { background:var(--panel); border:1px solid var(--border); color:var(--text);
                   padding:6px 14px; border-radius:6px; cursor:pointer; }
  .navrow button:disabled { opacity:.5; cursor:not-allowed; }
  .hint { text-align:center; color:var(--muted); font-size:12px; margin-top:8px; }
  .done { text-align:center; font-size:18px; color:var(--good); padding:40px 0; }
</style>
</head>
<body>

<header>
  <div class="left">
    <div class="counter"><span id="idx">0</span> / <span id="total">0</span></div>
    <div class="progress"><div id="bar"></div></div>
    <div class="kappa">κ vs judge: <span id="kappa">—</span></div>
  </div>
  <div>
    <button id="revealBtn" title="reveal LLM judge">显示 LLM 判断 (R)</button>
    <button id="downloadBtn">下载 JSON</button>
    <button id="resetBtn" title="clear saved labels">重置</button>
  </div>
</header>

<main>
  <div id="cardWrap"></div>
  <div class="hint">快捷键: 1 = 对, 0 / 2 = 错, S / Space = 跳过, ← = 上一题, → = 下一题, R = 显示 LLM 判断</div>
</main>

<script>
const DATA = __PAYLOAD__;
const STORAGE_KEY = "memarena_calibration_v2";
const METADATA = __METADATA__;

let labels = {};   // id -> 1 | 0 | "skip"
let cursor = 0;
let revealed = false;

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const s = JSON.parse(raw);
      labels = s.labels || {};
      cursor = Math.min(s.cursor || 0, DATA.length - 1);
    }
  } catch (e) {}
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify({labels, cursor}));
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s || "";
  return div.innerHTML;
}

function renderCard() {
  const wrap = document.getElementById("cardWrap");
  if (cursor >= DATA.length) {
    wrap.innerHTML = `<div class="card done">全部 ${DATA.length} 条已完成，请点击「下载 JSON」保存结果。</div>`;
    updateHeader();
    return;
  }
  const d = DATA[cursor];
  const existing = labels[d.id];
  const judgeCls = d.judge_correct ? "t" : "f";
  const judgeText = d.judge_correct ? "JUDGE: CORRECT (True)" : "JUDGE: INCORRECT (False)";
  const humanTag = existing === 1 ? "[你标记: 对]" :
                   existing === 0 ? "[你标记: 错]" :
                   existing === "skip" ? "[你标记: 跳过]" : "";
  wrap.innerHTML = `
    <div class="card">
      <div class="meta">
        <span class="badge">${escapeHtml(d.dim || '-')}</span>
        <span class="badge">${escapeHtml(d.cell || '-')}</span>
        <span class="badge">${escapeHtml(d.id || '-')}</span>
        <span style="flex:1"></span>
        <span style="color:#8b949e">${humanTag}</span>
      </div>
      <div class="block query">
        <div class="label">Query (用户问题)</div>
        <div class="body">${escapeHtml(d.query)}</div>
      </div>
      <div class="block gold">
        <div class="label">Gold (标准答案)</div>
        <div class="body">${escapeHtml(d.gold)}</div>
      </div>
      <div class="block pred">
        <div class="label">Model Prediction (模型输出)</div>
        <div class="body">${escapeHtml(d.prediction)}</div>
      </div>
      <div class="judge ${revealed ? 'shown' : ''}" id="judgeBox">
        <div class="verdict ${judgeCls}">${judgeText}</div>
        <div style="margin-top:6px">${escapeHtml(d.judge_reason)}</div>
      </div>
      <div class="controls">
        <button class="btn-good" onclick="mark(1)">✓ 对 (1)</button>
        <button class="btn-bad"  onclick="mark(0)">✗ 错 (0)</button>
        <button class="btn-skip" onclick="mark('skip')">↷ 跳过 (S)</button>
      </div>
      <div class="navrow">
        <button onclick="prev()" ${cursor === 0 ? 'disabled' : ''}>← 上一题</button>
        <div>${cursor + 1} / ${DATA.length}</div>
        <button onclick="next()" ${cursor >= DATA.length - 1 ? 'disabled' : ''}>下一题 →</button>
      </div>
    </div>
  `;
  updateHeader();
}

function updateHeader() {
  document.getElementById("idx").textContent = cursor;
  document.getElementById("total").textContent = DATA.length;
  const done = Object.values(labels).filter(v => v === 1 || v === 0).length;
  document.getElementById("bar").style.width = (100 * done / DATA.length) + "%";
  document.getElementById("kappa").textContent = computeKappa();
}

function computeKappa() {
  // Cohen's kappa between human labels (ignore skips) and DATA[*].judge_correct
  let n=0, a=0, b=0, both1=0, both0=0;
  for (const d of DATA) {
    const h = labels[d.id];
    if (h !== 1 && h !== 0) continue;
    const hj = d.judge_correct ? 1 : 0;
    n++;
    a += h;                 // human positives
    b += hj;                // judge positives
    if (h === 1 && hj === 1) both1++;
    if (h === 0 && hj === 0) both0++;
  }
  if (n < 5) return `— (n=${n})`;
  const po = (both1 + both0) / n;
  const pe = ((a/n) * (b/n)) + ((1 - a/n) * (1 - b/n));
  if (pe === 1) return `1.000 (n=${n})`;
  const k = (po - pe) / (1 - pe);
  return `${k.toFixed(3)} (n=${n})`;
}

function mark(val) {
  if (cursor >= DATA.length) return;
  const d = DATA[cursor];
  labels[d.id] = val;
  revealed = false;
  cursor = Math.min(cursor + 1, DATA.length);
  saveState();
  renderCard();
}

function next() {
  if (cursor < DATA.length - 1) {
    cursor++;
    revealed = false;
    saveState();
    renderCard();
  } else if (cursor === DATA.length - 1) {
    cursor++;
    saveState();
    renderCard();
  }
}

function prev() {
  if (cursor > 0) {
    cursor--;
    revealed = false;
    saveState();
    renderCard();
  }
}

function toggleReveal() {
  revealed = !revealed;
  const box = document.getElementById("judgeBox");
  if (box) box.classList.toggle("shown", revealed);
}

function download() {
  const out = {
    metadata: METADATA,
    generated_at: new Date().toISOString(),
    labels: labels,
    // Include the exact sample so downstream code can re-align by id
    sample: DATA.map(d => ({
      id: d.id, cell: d.cell, dim: d.dim, judge_correct: d.judge_correct
    })),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `human_calibration_labels_${new Date().toISOString().slice(0,10)}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function reset() {
  if (!confirm("清除所有已保存的标签并从头开始?")) return;
  labels = {};
  cursor = 0;
  revealed = false;
  saveState();
  renderCard();
}

document.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (e.key === "1" || e.key.toLowerCase() === "y") { mark(1); e.preventDefault(); }
  else if (e.key === "0" || e.key === "2" || e.key.toLowerCase() === "n") { mark(0); e.preventDefault(); }
  else if (e.key === "s" || e.key === "S" || e.key === " ") { mark("skip"); e.preventDefault(); }
  else if (e.key === "ArrowLeft") { prev(); e.preventDefault(); }
  else if (e.key === "ArrowRight") { next(); e.preventDefault(); }
  else if (e.key === "r" || e.key === "R") { toggleReveal(); e.preventDefault(); }
});

document.getElementById("revealBtn").addEventListener("click", toggleReveal);
document.getElementById("downloadBtn").addEventListener("click", download);
document.getElementById("resetBtn").addEventListener("click", reset);

loadState();
renderCard();
</script>

</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="MASim/runs/l_20260408_111046",
                    help="path to the l_<stamp> run directory")
    ap.add_argument("--judge", default="4omini", choices=["4omini", "qwen3"],
                    help="which judge's evaluation files to sample from")
    ap.add_argument("--n", type=int, default=500, help="number of instances to sample")
    ap.add_argument("--seed", type=int, default=20260419)
    ap.add_argument("--out", default="docs/human_calibration/index.html")
    ap.add_argument("--sample-json", default="docs/human_calibration/sample.json",
                    help="sidecar JSON to receive the exact 500-row sample payload + metadata; "
                         "downstream code joins user's label JSON against this for kappa analysis")
    ap.add_argument("--include-non-judge", action="store_true",
                    help="include rows whose scoring_method is not 'judge' "
                         "(default: excluded; paper scopes human calibration "
                         "to the LLM-judge-scored subset only — see "
                         "paper/chapters/11_appendix_placeholders.tex:44-45)")
    ap.add_argument("--pin-labels-json", default=None,
                    help="path to a previously downloaded labels JSON; any "
                         "instance_id with a recorded label (including 'skip') "
                         "is pinned into the new sample so progress is preserved")
    args = ap.parse_args()

    root = Path(args.run_dir).resolve()
    if not root.exists():
        raise SystemExit(f"run dir not found: {root}")

    files = discover_eval_files(root, args.judge)
    if not files:
        raise SystemExit(f"no evaluation files under {root} for judge={args.judge}")
    print(f"found {len(files)} evaluation files", file=sys.stderr)

    by_cell = load_rows(files)
    total_rows = sum(len(r) for r in by_cell.values())
    print(f"loaded {total_rows} rows across {len(by_cell)} cells", file=sys.stderr)

    judge_only = not args.include_non_judge
    pool = filter_pool(by_cell, judge_only=judge_only)
    print(f"pool after filtering: {len(pool)} rows "
          f"(judge_only={judge_only})",
          file=sys.stderr)

    pinned_ids = set()
    if args.pin_labels_json:
        backup = json.loads(Path(args.pin_labels_json).read_text())
        pinned_ids = set(backup.get("labels", {}).keys())
        print(f"pinning {len(pinned_ids)} already-judged IDs from "
              f"{args.pin_labels_json}", file=sys.stderr)
        # Recover any pinned rows that the filter excluded (rule-based / no-gold):
        # the user has already judged them, so preserve regardless.
        full_pool = build_full_pool(by_cell)
        pool_ids = {r.get("instance_id") for (_c, r) in pool.values()}
        recovered = 0
        for key, (cell, r) in full_pool.items():
            iid = r.get("instance_id")
            if iid in pinned_ids and iid not in pool_ids:
                # Ensure judge verdict exists so downstream code works
                jc = r.get("judge_correct")
                if jc is None:
                    jc = r.get("correct")
                if jc is None:
                    continue
                pool[key] = (cell, r)
                recovered += 1
        if recovered:
            print(f"  recovered {recovered} pinned rows that the filter excluded",
                  file=sys.stderr)
        still_missing = pinned_ids - {r.get("instance_id") for (_c, r) in pool.values()}
        if still_missing:
            print(f"  WARNING: {len(still_missing)} pinned IDs not found in any "
                  f"eval file: {sorted(still_missing)[:5]}"
                  f"{'...' if len(still_missing) > 5 else ''}",
                  file=sys.stderr)
            pinned_ids -= still_missing

    rng = random.Random(args.seed)
    sample = stratified_sample(pool, args.n, rng, pinned_ids=pinned_ids)
    print(f"sampled {len(sample)} rows", file=sys.stderr)

    payload = build_payload(sample)

    metadata = {
        "run_dir": str(root),
        "judge": args.judge,
        "n": args.n,
        "seed": args.seed,
        "n_files": len(files),
        "n_rows_raw": total_rows,
        "n_rows_pool": len(pool),
        "include_non_judge": bool(args.include_non_judge),
        "filter_rule": (
            "scoring_method == 'judge' only (paper Appendix L scope)"
            if judge_only else "no scoring_method filter"
        ),
        "pinned_ids_count": len(pinned_ids),
        "stratification": "dimension x judge_correct",
    }

    html_text = HTML_TEMPLATE \
        .replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False)) \
        .replace("__METADATA__", json.dumps(metadata, ensure_ascii=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size/1024:.1f} KB)", file=sys.stderr)

    if args.sample_json:
        json_path = Path(args.sample_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps({"metadata": metadata, "sample": payload},
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"wrote {json_path} ({json_path.stat().st_size/1024:.1f} KB)",
              file=sys.stderr)


if __name__ == "__main__":
    main()
