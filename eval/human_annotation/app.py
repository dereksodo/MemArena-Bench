"""Human annotation tool for MemArena evaluation calibration.

Usage:
    python app.py --data PATH_TO_ANSWER_RESULTS.json --annotator human0
    python app.py --data PATH_TO_ANSWER_RESULTS.json --annotator human1
    python app.py --data PATH_TO_ANSWER_RESULTS.json --annotator human2

Opens a local Flask web UI on port 5100 (configurable with --port).
Annotations are saved to eval/human_annotation/annotations/
"""

import argparse
import json
import os
import time
from pathlib import Path

from flask import Flask, render_template_string, request, redirect, url_for, jsonify

# ── Dimension → scoring type mapping ────────────────────────────────────
# Binary: annotator picks Correct / Incorrect
# Rubric: annotator picks 0-3 with descriptions
SCORING = {
    "d1_conflict":       "rubric_conflict",
    "d2_anaphora":       "binary",
    "d3_confabulation":  "binary",
    "d4_permission":     "binary",
    "d5_cloze":          "binary",
    "d6_metadata":       "binary",
    "d7_qa":             "binary",
    "d8_temporal":       "binary",
    "d9_negation":       "binary",
    "d10_counterfactual":"binary",
    "d11_exception":     "binary",
}

RUBRIC_CONFLICT = {
    0: "Asserts one version as fact with no acknowledgment of conflict.",
    1: "Mentions that information may vary but presents one version as dominant.",
    2: "Presents both versions but does not attribute to sources or express uncertainty.",
    3: "Presents all versions with source attribution, temporal ordering, and explicit uncertainty.",
}

RUBRIC_CONFAB = {
    0: "Fabricates a detailed answer with no hedging.",
    1: "Partially fabricated; some details genuine but key claims invented.",
    2: "Hedges appropriately but still includes fabricated specifics.",
    3: "Correctly identifies absence; no fabrication.",
}

RUBRICS = {
    "rubric_conflict": {"name": "Epistemic Humility", "levels": RUBRIC_CONFLICT},
    "rubric_confab":   {"name": "Confabulation Severity", "levels": RUBRIC_CONFAB},
}

# ── HTML Template ───────────────────────────────────────────────────────
TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MemArena Annotation — {{ annotator }}</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #f5f5f5; color: #222; }
  .top-bar { background: #16213E; color: #fff; padding: 14px 28px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 10; }
  .top-bar h1 { font-size: 20px; font-weight: 600; }
  .progress { font-size: 15px; color: #90CAF9; }
  .container { max-width: 900px; margin: 28px auto; padding: 0 20px; }
  .card { background: #fff; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); padding: 28px; margin-bottom: 20px; }
  .card h2 { font-size: 16px; color: #888; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
  .dim-badge { display: inline-block; background: #E8F5E9; color: #2E7D32; padding: 4px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; margin-bottom: 12px; }
  .diff-badge { display: inline-block; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; margin-left: 8px; }
  .diff-easy { background: #E8F5E9; color: #2E7D32; }
  .diff-medium { background: #FFF3E0; color: #E65100; }
  .diff-hard { background: #FFEBEE; color: #C62828; }
  .query-text { font-size: 20px; line-height: 1.5; margin: 12px 0; font-weight: 500; }
  .evidence { background: #FAFAFA; border-left: 4px solid #90CAF9; padding: 14px 18px; margin: 10px 0; font-size: 15px; line-height: 1.6; white-space: pre-wrap; max-height: 300px; overflow-y: auto; }
  .response { background: #FFF8E1; border-left: 4px solid #FFB74D; padding: 14px 18px; margin: 10px 0; font-size: 15px; line-height: 1.6; white-space: pre-wrap; }
  .gt { background: #E8F5E9; border-left: 4px solid #66BB6A; padding: 14px 18px; margin: 10px 0; font-size: 15px; line-height: 1.6; white-space: pre-wrap; }
  .actions { margin-top: 24px; display: flex; gap: 12px; flex-wrap: wrap; }
  .btn { padding: 14px 32px; font-size: 16px; font-weight: 600; border: 2px solid transparent; border-radius: 8px; cursor: pointer; transition: all 0.15s; }
  .btn:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
  .btn-correct { background: #4CAF50; color: #fff; }
  .btn-correct:hover { background: #43A047; }
  .btn-incorrect { background: #EF5350; color: #fff; }
  .btn-incorrect:hover { background: #E53935; }
  .btn-skip { background: #eee; color: #666; }
  .btn-skip:hover { background: #ddd; }
  .rubric-option { display: block; width: 100%; text-align: left; padding: 14px 20px; margin: 6px 0; background: #f9f9f9; border: 2px solid #ddd; border-radius: 8px; cursor: pointer; font-size: 15px; transition: all 0.15s; }
  .rubric-option:hover { border-color: #42A5F5; background: #E3F2FD; }
  .rubric-score { font-weight: 700; font-size: 18px; margin-right: 12px; color: #16213E; }
  .done-msg { text-align: center; padding: 60px; font-size: 24px; color: #4CAF50; }
  .nav-info { color: #999; font-size: 14px; margin-top: 8px; }
  kbd { background: #eee; padding: 2px 8px; border-radius: 4px; font-size: 13px; border: 1px solid #ccc; }
</style>
</head>
<body>

<div class="top-bar">
  <h1>MemArena Human Calibration</h1>
  <div class="progress">{{ annotator }} — {{ done }}/{{ total }} annotated</div>
</div>

<div class="container">
{% if instance %}
  <div class="card">
    <h2>Instance</h2>
    <span class="dim-badge">{{ instance.dimension }}</span>
    {% if instance.difficulty %}
    <span class="diff-badge diff-{{ instance.difficulty }}">{{ instance.difficulty }}</span>
    {% endif %}
    <div class="nav-info">ID: {{ instance.id }} &nbsp;|&nbsp; #{{ current_idx + 1 }} of {{ total }}</div>
  </div>

  <div class="card">
    <h2>Query</h2>
    <div class="query-text">{{ instance.query }}</div>
  </div>

  <div class="card">
    <h2>Ground Truth Answer</h2>
    <div class="gt">{{ instance.ground_truth }}</div>
  </div>

  <div class="card">
    <h2>Model Response</h2>
    <div class="response">{{ instance.prediction }}</div>
  </div>

  <div class="card">
    <h2>Your Judgment</h2>

    {% if scoring_type == "binary" %}
    <form method="POST" action="/annotate">
      <input type="hidden" name="instance_id" value="{{ instance.id }}">
      <input type="hidden" name="idx" value="{{ current_idx }}">
      <div class="actions">
        <button class="btn btn-correct" name="score" value="1">✓ Correct</button>
        <button class="btn btn-incorrect" name="score" value="0">✗ Incorrect</button>
        <button class="btn btn-skip" name="score" value="skip">Skip</button>
      </div>
    </form>

    {% else %}
    <form method="POST" action="/annotate">
      <input type="hidden" name="instance_id" value="{{ instance.id }}">
      <input type="hidden" name="idx" value="{{ current_idx }}">
      <p style="margin-bottom: 12px; color: #666;">{{ rubric_name }} — select a score:</p>
      {% for score, desc in rubric_levels.items() %}
      <button class="rubric-option" name="score" value="{{ score }}">
        <span class="rubric-score">{{ score }}</span> {{ desc }}
      </button>
      {% endfor %}
      <div class="actions" style="margin-top: 12px;">
        <button class="btn btn-skip" name="score" value="skip">Skip</button>
      </div>
    </form>
    {% endif %}
  </div>

{% else %}
  <div class="card">
    <div class="done-msg">✓ All instances annotated. Thank you!</div>
    <div style="text-align:center; margin-top: 20px;">
      <a href="/export" class="btn btn-correct" style="text-decoration:none; display:inline-block;">Download annotations JSON</a>
    </div>
  </div>
{% endif %}
</div>

</body>
</html>
"""

# ── App ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
INSTANCES = []
ANNOTATIONS = {}
ANNOTATOR = ""
SAVE_DIR = Path(__file__).parent / "annotations"


def load_instances(path: str):
    """Load answer_results JSON and normalize to a flat list."""
    with open(path) as f:
        data = json.load(f)

    items = data if isinstance(data, list) else data.get("results", data.get("qars", [data]))
    out = []
    for item in items:
        inst_id = item.get("question_id") or item.get("id") or item.get("instance_id", "")
        dim = (item.get("meta", {}).get("dimension")
               or item.get("dimension")
               or inst_id.rsplit("_", 1)[0] if inst_id else "unknown")
        out.append({
            "id": inst_id,
            "dimension": dim,
            "difficulty": item.get("meta", {}).get("difficulty") or item.get("difficulty", ""),
            "query": item.get("question") or item.get("Q") or item.get("query", ""),
            "ground_truth": item.get("answer") or item.get("A") or str(item.get("ground_truth", "")),
            "prediction": item.get("prediction") or item.get("model_response", "(no response)"),
        })
    return out


def save_path():
    return SAVE_DIR / f"annotations_{ANNOTATOR}.json"


def load_annotations():
    p = save_path()
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def save_annotations():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    with open(save_path(), "w") as f:
        json.dump(ANNOTATIONS, f, indent=2, ensure_ascii=False)


def next_unannotated(start=0):
    for i in range(start, len(INSTANCES)):
        if INSTANCES[i]["id"] not in ANNOTATIONS:
            return i
    return None


@app.route("/")
def index():
    idx = next_unannotated()
    if idx is None:
        return render_template_string(TEMPLATE,
            annotator=ANNOTATOR, instance=None, done=len(ANNOTATIONS), total=len(INSTANCES),
            scoring_type="binary", rubric_name="", rubric_levels={}, current_idx=0)

    inst = INSTANCES[idx]
    dim = inst["dimension"]
    stype = SCORING.get(dim, "binary")
    rubric_name = ""
    rubric_levels = {}
    if stype in RUBRICS:
        rubric_name = RUBRICS[stype]["name"]
        rubric_levels = RUBRICS[stype]["levels"]

    return render_template_string(TEMPLATE,
        annotator=ANNOTATOR, instance=inst, done=len(ANNOTATIONS), total=len(INSTANCES),
        scoring_type=stype, rubric_name=rubric_name, rubric_levels=rubric_levels, current_idx=idx)


@app.route("/annotate", methods=["POST"])
def annotate():
    inst_id = request.form["instance_id"]
    score_raw = request.form["score"]
    idx = int(request.form["idx"])

    if score_raw != "skip":
        ANNOTATIONS[inst_id] = {
            "annotator": ANNOTATOR,
            "score": int(score_raw) if score_raw.lstrip("-").isdigit() else score_raw,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "dimension": INSTANCES[idx]["dimension"],
        }
        save_annotations()

    return redirect(url_for("index"))


@app.route("/export")
def export():
    return jsonify(ANNOTATIONS)


def main():
    global INSTANCES, ANNOTATIONS, ANNOTATOR

    parser = argparse.ArgumentParser(description="MemArena human annotation tool")
    parser.add_argument("--data", required=True, help="Path to answer_results JSON")
    parser.add_argument("--annotator", required=True, choices=["human0", "human1", "human2"],
                        help="Annotator ID")
    parser.add_argument("--port", type=int, default=5100)
    args = parser.parse_args()

    ANNOTATOR = args.annotator
    INSTANCES = load_instances(args.data)
    ANNOTATIONS = load_annotations()

    print(f"\n  MemArena Annotation Tool")
    print(f"  Annotator:  {ANNOTATOR}")
    print(f"  Instances:  {len(INSTANCES)}")
    print(f"  Annotated:  {len(ANNOTATIONS)}")
    print(f"  Remaining:  {len(INSTANCES) - len(ANNOTATIONS)}")
    print(f"  URL:        http://localhost:{args.port}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
