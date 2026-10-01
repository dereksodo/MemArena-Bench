"""Phase 3: full D6 rerun on the 75 main_5x5x3 cells using the new 5-label rubric.

Reads predictions from existing ``evaluation_results_*.json`` files (the
``prediction`` field), runs the new judge, writes back updated label / reason /
correctness / fact_in_output for every d4_perm_* record. Backs up the
original file to ``*_legacy.json`` once per cell on first touch.

Inputs:
  - ``experiments_index.csv`` resolved via paper_data.load_all_cells(
      include_ablation=False).
  - ``OPENROUTER_API_KEY`` / ``OPENROUTER_BASE_URL`` from .env.

Concurrency: ``--workers`` judge requests in flight (default 128), shared by
all cells, so a single ``--eval-path`` cell is relabelled in parallel too (it
used to be one thread per cell, i.e. one request at a time). Each request
retries up to 3x with a short backoff. Empty predictions are short-circuited
(no API call) per ``_score_d4_armB``.

Judge: ``--judge-model`` (default openai/gpt-4o-mini-2024-07-18, the paper's).
DeepSeek models are called with reasoning on and a 4096-token budget
(``memarena.judges.judge_request_kwargs``); prompts, temperature 0 and parsing
are the same for every judge.

Idempotency: if a record already has the new vocabulary in ``policy_category``
(one of D6_LABELS) and a non-empty ``rationale_v2``, skip it. So the script
is safe to rerun on partial output.

Run:
    python3 scripts/rerun_d6_judge.py [--max-cells N] [--dry-run]
    python3 scripts/rerun_d6_judge.py --eval-path <evaluation_results_*.json>   # one cell
    python3 scripts/rerun_d6_judge.py --eval-path <...> --judge-model deepseek/deepseek-v4.1-flash
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_env() -> dict[str, str]:
    env_path = REPO_ROOT / ".env"
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


DEFAULT_JUDGE_MODEL = "openai/gpt-4o-mini-2024-07-18"
REQUEST_TIMEOUT_S = 60    # a DeepSeek verdict takes seconds; a reply still
                          # reasoning after a minute is stuck, so resend it


def _make_client(workers: int = 16):
    env = _load_env()
    # Environment first, then .env: the same precedence as scripts/llmjudge.py.
    api_key = os.getenv("OPENROUTER_API_KEY") or env.get("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL") or env.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env or env vars")
    from openai import OpenAI
    kwargs: dict = {"api_key": api_key, "base_url": base_url,
                    # the SDK default is a 600 s read timeout: one stuck request
                    # would hold a worker for ten minutes
                    "timeout": REQUEST_TIMEOUT_S, "max_retries": 1}
    try:
        import httpx
        from openai import DefaultHttpxClient
        n = max(1, int(workers))
        kwargs["http_client"] = DefaultHttpxClient(
            limits=httpx.Limits(max_connections=max(1000, n), max_keepalive_connections=n),
            timeout=REQUEST_TIMEOUT_S,
        )
    except ImportError:
        pass
    return OpenAI(**kwargs)


def _judge_record(
    client: Any,
    model: str,
    rec: dict,
    gold_fact: str,
    max_retries: int = 5,
    *,
    question: Optional[str] = None,
    gt: Optional[dict] = None,
) -> tuple[str, bool, str, str]:
    """Run new D6 judge on one record. Returns (label, fact_in_output,
    rationale, reason_token). Empty prediction is short-circuited to
    OTHER without API call.

    ``question`` / ``gt`` are the instance's query text and ground truth; the
    judge needs both (whose fact it is lives in ``gt["target_agent"]``).
    Without them it falls back to the record's question id and bare gold."""
    from eval.src.scoring import (
        _ARMB_JUDGE_SYSTEM,
        _build_armB_user,
        _parse_armB_response,
        _strip_thinking,
        _armB_correctness,
    )
    from memarena.judges import judge_create, judge_request_kwargs
    raw_pred = rec.get("prediction") or ""
    pred = _strip_thinking(raw_pred.strip())
    expected_mode = str(rec.get("expected_answer_mode") or "").lower()

    if not pred:
        ok, reason = _armB_correctness("OTHER", expected_mode)
        return "OTHER", False, "", reason

    user_msg = _build_armB_user(
        prediction=pred,
        question=question if question is not None else rec.get("question_id", ""),
        gt=dict(gt) if gt else {"fact": gold_fact},
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = judge_create(client, 
                model=model,
                messages=[
                    {"role": "system", "content": _ARMB_JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                **judge_request_kwargs(model),
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_armB_response(raw)
            label = parsed["category"]
            if label == "PARSE_ERROR":
                # an empty or non-JSON reply (seen with reasoning models): resend
                raise ValueError(f"unparseable judge reply: {raw[:120]!r}")
            fact_in = bool(parsed["leaked_fact_in_output"])
            rationale = parsed["rationale"]
            ok, reason = _armB_correctness(label, expected_mode)
            return label, fact_in, rationale, reason
        except Exception as e:
            last_err = e
            if attempt + 1 < max_retries:  # no pointless sleep after the last try
                time.sleep(1.5 ** attempt)
    # All retries failed
    ok, reason = _armB_correctness("PARSE_ERROR", expected_mode)
    return "PARSE_ERROR", False, f"judge_error: {last_err}"[:200], reason


@dataclass
class CellStats:
    cell_key: tuple
    n_total: int = 0
    n_done: int = 0
    n_skipped: int = 0
    n_error: int = 0
    label_counts: dict = None

    def __post_init__(self):
        self.label_counts = {}


def process_cell(
    json_path: Path,
    cell_key: tuple,
    client: Any,
    model: str,
    dry_run: bool = False,
    executor: Optional[Executor] = None,
    instances: Optional[dict] = None,
) -> CellStats:
    """Re-judge all d4_perm_* records in one evaluation_results JSON file.
    Backs up original to *_legacy.json on first touch. Idempotent.

    With ``executor`` the judge calls of this cell run on it concurrently;
    without, one after another."""
    from eval.src.scoring import D6_LABELS, _armB_correctness

    stats = CellStats(cell_key=cell_key)

    backup_path = json_path.with_name(json_path.stem + "_legacy.json")
    data = json.loads(json_path.read_text())
    details = data.get("details", [])

    # Identify d4_perm records
    d4_records = [
        (i, d) for i, d in enumerate(details)
        if str(d.get("question_id", "")).startswith("d4_perm")
    ]
    stats.n_total = len(d4_records)
    if not d4_records:
        return stats

    # First-touch backup (only if not already backed up)
    if not backup_path.exists() and not dry_run:
        shutil.copyfile(json_path, backup_path)

    todo = []
    for _i, rec in d4_records:
        # Idempotency: skip if already in new vocabulary
        existing_cat = str(rec.get("policy_category") or "").upper()
        if existing_cat in D6_LABELS and rec.get("rationale_v2"):
            stats.n_skipped += 1
            stats.label_counts[existing_cat] = stats.label_counts.get(existing_cat, 0) + 1
            continue
        todo.append(rec)

    def judge(rec: dict) -> tuple[str, bool, str, str]:
        if dry_run:
            return "DRY_RUN", False, "", ""
        inst = (instances or {}).get(rec.get("question_id"))
        if instances is not None and inst is None:
            raise KeyError(f"{rec.get('question_id')} is not in the D6 instance file")
        return _judge_record(client, model, rec, str(rec.get("gold_answer") or ""),
                             question=inst["query"] if inst else None,
                             gt=inst["ground_truth"] if inst else None)

    if executor is not None and not dry_run:
        results = list(executor.map(judge, todo))
    else:
        results = [judge(rec) for rec in todo]

    for rec, (label, fact_in, rationale, _reason) in zip(todo, results):
        expected_mode = str(rec.get("expected_answer_mode") or "").lower()
        ok, reason = _armB_correctness(label, expected_mode)
        # Update record in place
        if not dry_run:
            rec["policy_category"] = label  # new value vocabulary
            rec["leaked_fact_in_output"] = fact_in
            rec["rationale_v2"] = rationale  # new diagnostic field
            rec["reason"] = reason  # update reason token to new vocabulary
            rec["correct"] = ok
            rec["score"] = 1.0 if ok else 0.0
        stats.n_done += 1
        if label == "PARSE_ERROR":
            stats.n_error += 1
        stats.label_counts[label] = stats.label_counts.get(label, 0) + 1

    if not dry_run:
        _refresh_summary(data)
        data.setdefault("summary", {})["d6_relabel_judge_model"] = model
        json_path.write_text(json.dumps(data, indent=2))

    return stats


def _refresh_summary(data: dict) -> None:
    """Recompute the accuracy fields of ``summary`` from ``details``.

    The relabel changes ``correct`` / ``score`` of D6 records in ``details``
    only; without this the file's summary.accuracy keeps the step-1 scoring
    and disagrees with the mean of details[].correct.
    """
    details = data.get("details") or []
    summary = data.setdefault("summary", {})
    scored = [d for d in details if d.get("answer_scored", True)]
    n = len(scored)
    correct = sum(1 for d in scored if d.get("correct"))
    d4 = [d for d in scored if d.get("policy_category") is not None]
    summary.update({
        "answer_scored_total": n,
        "answer_scored_correct": correct,
        "total": n,
        "correct": correct,
        "accuracy": correct / n if n else 0.0,
        "mean_score": (sum(float(d.get("score") or 0.0) for d in scored) / n) if n else 0.0,
        "policy_total": len(d4),
        "policy_accuracy": (sum(1 for d in d4 if d.get("correct")) / len(d4)) if d4 else 0.0,
        "summary_refreshed_by": "scripts/rerun_d6_judge.py",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Limit to first N cells (for debugging)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't call API, don't write files")
    parser.add_argument("--workers", type=int, default=128,
                        help="judge requests in flight, shared by all cells (default 128)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help=f"judge model (default {DEFAULT_JUDGE_MODEL}; DeepSeek models run with "
                             "reasoning on, 4096 tokens)")
    parser.add_argument("--eval-path", type=Path, action="append", default=None,
                        help="Relabel only this evaluation_results_*.json (repeatable) "
                             "instead of the cells in experiments_index.csv. Used by "
                             "scripts/judge_cell.sh.")
    parser.add_argument("--dataset", type=Path,
                        default=Path(os.getenv("MEMARENA_DATASET_DIR") or REPO_ROOT / "data" / "benchmark"),
                        help="benchmark dir whose eval_instances/d4_permission.jsonl gives each "
                             "record's question text and ground truth (default $MEMARENA_DATASET_DIR)")
    args = parser.parse_args()

    if args.eval_path:
        cells = [((str(p),), SimpleNamespace(source_path=p.resolve())) for p in args.eval_path]
    else:
        from memarena.figures.paper_data import load_all_cells
        grid = load_all_cells(include_ablation=True)
        cells = sorted(grid.items())
    if args.max_cells:
        cells = cells[: args.max_cells]
    model = args.judge_model
    workers = max(1, int(args.workers))
    print(f"Re-judging {len(cells)} cells via {model} (workers={workers}, dry_run={args.dry_run})")

    from eval.src.permission_metrics import load_instances
    instances = load_instances(args.dataset / "eval_instances" / "d4_permission.jsonl")

    if args.dry_run:
        client = None
    else:
        client = _make_client(workers)

    t0 = time.time()
    overall = {
        "cells_done": 0, "records_done": 0, "records_skipped": 0,
        "records_error": 0, "label_counts": {},
    }

    # Requests of every cell share one pool of ``workers`` threads; a few
    # cells are read and written at a time (their threads only wait).
    request_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="d6judge")
    with request_pool, ThreadPoolExecutor(max_workers=min(4, max(1, len(cells)))) as executor:
        futures = {
            executor.submit(process_cell, cell.source_path, key, client, model, args.dry_run, request_pool,
                            instances): key
            for key, cell in cells
        }
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                s: CellStats = fut.result()
            except Exception as e:
                print(f"  cell {key} FAILED with {type(e).__name__}: {e}")
                continue
            overall["cells_done"] += 1
            overall["records_done"] += s.n_done
            overall["records_skipped"] += s.n_skipped
            overall["records_error"] += s.n_error
            for k, v in (s.label_counts or {}).items():
                overall["label_counts"][k] = overall["label_counts"].get(k, 0) + v

            elapsed = time.time() - t0
            rate = overall["records_done"] / elapsed if elapsed > 0 else 0
            est_total = (overall["records_done"] + (len(cells) - overall["cells_done"]) * 200)
            eta_s = (est_total - overall["records_done"]) / rate if rate > 0 else 0
            print(
                f"[{overall['cells_done']:>2}/{len(cells)}] {key}  "
                f"+{s.n_done}done +{s.n_skipped}skip +{s.n_error}err  "
                f"labels={s.label_counts}  "
                f"elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta_s:.0f}s"
            )

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"DONE in {elapsed:.0f}s")
    print(f"  cells_done={overall['cells_done']}/{len(cells)}")
    print(f"  records_done={overall['records_done']}")
    print(f"  records_skipped={overall['records_skipped']}")
    print(f"  records_error={overall['records_error']}")
    print(f"  label_counts={overall['label_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
