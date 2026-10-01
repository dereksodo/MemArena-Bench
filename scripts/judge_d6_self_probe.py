"""Phase 3: judge the D6 self-probe answer files using the existing 5-label rubric.

Mirrors ``scripts/rerun_d6_judge.py`` but operates on a user-supplied
directory tree instead of ``experiments_index.csv``. Self-probe answer
files are produced by running ``eval/cli.py --d6-probe-mode self_ego ...``
on H100 and live in their own directory tree (recommended path:
``out/d6_self_probe/``).

Inputs:
  - ``--input-root``: directory containing self-probe output files (recursed).
    Accepts both:
      * ``evaluation_results_*.json`` (dict with "details" key) — legacy format
      * ``answer_results_run.json`` (list) — produced by H100 inference script;
        metadata loaded from sibling ``masim_qa_run.json``;
        judged output written to sibling ``evaluation_results_run.json``.
    Default: ``out/d6_self_probe``.
  - ``OPENROUTER_API_KEY`` / ``OPENROUTER_BASE_URL`` from ``.env``.

Concurrency: 16 worker threads, retry up to 3x with backoff. Empty
predictions are short-circuited (no API call) per ``_score_d4_armB``.

Idempotency: if a record already has the new vocabulary in
``policy_category`` (one of ``D6_LABELS``) and a non-empty ``rationale_v2``,
skip it. Safe to rerun on partial output.

Run on H100 (after the self-probe inference completes):
    OPENROUTER_API_KEY=sk-... python3 scripts/judge_d6_self_probe.py
or with custom path:
    python3 scripts/judge_d6_self_probe.py --input-root /path/to/out
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _make_client():
    env = _load_env()
    api_key = env.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    base_url = (
        env.get("OPENROUTER_BASE_URL")
        or os.getenv("OPENROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    )
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not found in .env or env vars")
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


def _judge_record(
    client: Any,
    model: str,
    rec: dict,
    gold_fact: str,
    max_retries: int = 3,
) -> tuple[str, bool, str, str]:
    """Run new D6 judge on one record. Empty prediction is short-circuited."""
    from eval.src.scoring import (
        _ARMB_JUDGE_SYSTEM,
        _build_armB_user,
        _parse_armB_response,
        _strip_thinking,
        _armB_correctness,
    )
    raw_pred = rec.get("prediction") or ""
    pred = _strip_thinking(raw_pred.strip())
    expected_mode = str(rec.get("expected_answer_mode") or "").lower()

    if not pred:
        ok, reason = _armB_correctness("OTHER", expected_mode)
        return "OTHER", False, "", reason

    user_msg = _build_armB_user(
        prediction=pred,
        question=rec.get("question_id", ""),
        gt={"fact": gold_fact},
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _ARMB_JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=200,
            )
            raw = resp.choices[0].message.content or ""
            parsed = _parse_armB_response(raw)
            label = parsed["category"]
            fact_in = bool(parsed["leaked_fact_in_output"])
            rationale = parsed["rationale"]
            ok, reason = _armB_correctness(label, expected_mode)
            return label, fact_in, rationale, reason
        except Exception as e:
            last_err = e
            time.sleep(1.5 ** attempt)
    ok, reason = _armB_correctness("PARSE_ERROR", expected_mode)
    return "PARSE_ERROR", False, f"judge_error: {last_err}"[:200], reason


@dataclass
class CellStats:
    json_path: Path
    n_total: int = 0
    n_done: int = 0
    n_skipped: int = 0
    n_error: int = 0
    label_counts: dict = None

    def __post_init__(self):
        self.label_counts = {}


def _load_masim_qa_lookup(answer_results_path: Path) -> dict[str, dict]:
    """Load masim_qa_run.json from the same directory, return dict keyed by question_id."""
    masim_path = answer_results_path.parent / "masim_qa_run.json"
    if not masim_path.exists():
        return {}
    try:
        data = json.loads(masim_path.read_text())
        qars = data.get("qars", [])
        return {q["id"]: q for q in qars if "id" in q}
    except Exception:
        return {}


def _answer_results_to_eval_format(
    answer_list: list[dict],
    masim_lookup: dict[str, dict],
) -> list[dict]:
    """Convert answer_results list records into evaluation_results detail format."""
    out = []
    for rec in answer_list:
        qid = rec.get("question_id", "")
        masim = masim_lookup.get(qid, {})
        meta = masim.get("meta", {})
        detail = {
            "question_id": qid,
            "question": rec.get("question", masim.get("Q", "")),
            "gold_answer": rec.get("answer", masim.get("A", "")),
            "prediction": rec.get("prediction", ""),
            "model": rec.get("model", ""),
            "expected_answer_mode": meta.get("expected_answer_mode", ""),
            "policy_expected": meta.get("policy_expected", ""),
            "answer_time_ms": rec.get("answer_time_ms"),
            "prompt_tokens": rec.get("prompt_tokens"),
            "completion_tokens": rec.get("completion_tokens"),
        }
        out.append(detail)
    return out


def process_file(
    json_path: Path,
    client: Any,
    model: str,
    dry_run: bool = False,
) -> CellStats:
    """Re-judge all d4_perm_* records in one self-probe output file.

    Accepts two input formats:
    - evaluation_results_*.json  (dict with "details" list): judged in-place
    - answer_results_run.json    (list): metadata merged from masim_qa_run.json,
      judged output written to sibling evaluation_results_run.json
    """
    from eval.src.scoring import D6_LABELS, _armB_correctness

    stats = CellStats(json_path=json_path)

    try:
        raw = json.loads(json_path.read_text())
    except Exception as exc:
        print(f"  SKIP unreadable file {json_path}: {exc}")
        return stats

    # --- Detect format and build working data ---
    is_answer_results = isinstance(raw, list)
    if is_answer_results:
        masim_lookup = _load_masim_qa_lookup(json_path)
        details = _answer_results_to_eval_format(raw, masim_lookup)
        data: dict = {"details": details, "_source": "answer_results_run.json"}
        out_path = json_path.parent / "evaluation_results_run.json"
        # If output already exists, load it so we can resume idempotently
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text())
                if isinstance(existing, dict) and "details" in existing:
                    data = existing
                    details = data["details"]
            except Exception:
                pass
    else:
        data = raw
        details = data.get("details", []) if isinstance(data, dict) else []
        out_path = json_path
        backup_path = json_path.with_name(json_path.stem + "_legacy.json")
        if not backup_path.exists() and not dry_run and details:
            shutil.copyfile(json_path, backup_path)

    d4_records = [
        (i, d) for i, d in enumerate(details)
        if str(d.get("question_id", "")).startswith("d4_perm")
    ]
    stats.n_total = len(d4_records)
    if not d4_records:
        return stats

    for i, rec in d4_records:
        existing_cat = str(rec.get("policy_category") or "").upper()
        if existing_cat in D6_LABELS and rec.get("rationale_v2"):
            stats.n_skipped += 1
            stats.label_counts[existing_cat] = stats.label_counts.get(existing_cat, 0) + 1
            continue

        gold_fact = str(rec.get("gold_answer") or "")
        if dry_run:
            label, fact_in, rationale, reason = "DRY_RUN", False, "", ""
        else:
            label, fact_in, rationale, reason = _judge_record(client, model, rec, gold_fact)

        expected_mode = str(rec.get("expected_answer_mode") or "").lower()
        ok, _reason = _armB_correctness(label, expected_mode)

        if not dry_run:
            rec["policy_category"] = label
            rec["leaked_fact_in_output"] = fact_in
            rec["rationale_v2"] = rationale
            rec["reason"] = reason
            rec["correct"] = ok
            rec["score"] = 1.0 if ok else 0.0
            rec["d6_probe_mode"] = "self_ego"

        stats.n_done += 1
        if label == "PARSE_ERROR":
            stats.n_error += 1
        stats.label_counts[label] = stats.label_counts.get(label, 0) + 1

    if not dry_run:
        out_path.write_text(json.dumps(data, indent=2))

    return stats


def discover_files(input_root: Path) -> list[Path]:
    """Find self-probe output files under input_root (recursed).

    Discovers two kinds:
    - evaluation_results_*.json  (dict+details): judged in-place. Skips *_legacy.
    - answer_results_run.json    (list): metadata merged from masim_qa_run.json,
      output written to sibling evaluation_results_run.json.
      Skipped if a sibling evaluation_results_run.json already exists (use that
      file directly instead, so we don't double-count the same directory).
    """
    out: list[Path] = []
    eval_dirs: set[Path] = set()

    for p in sorted(input_root.rglob("evaluation_results_*.json")):
        if p.stem.endswith("_legacy"):
            continue
        if "/runs/" in p.as_posix():
            continue
        out.append(p)
        eval_dirs.add(p.parent)

    for p in sorted(input_root.rglob("answer_results_run.json")):
        if "/runs/" in p.as_posix():
            continue
        if p.parent not in eval_dirs:
            out.append(p)

    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path,
        default=REPO_ROOT / "out" / "d6_self_probe",
        help="Directory containing self-probe evaluation_results JSON files.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit to first N files (for debugging)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    files = discover_files(args.input_root)
    if args.max_files:
        files = files[: args.max_files]

    if not files:
        print(f"No self-probe output files found under {args.input_root}")
        print("Expected: evaluation_results_*.json  OR  answer_results_run.json")
        print("Did the H100 self-probe inference run produce output here?")
        return 1

    print(f"Re-judging {len(files)} self-probe files via openai/gpt-4o-mini-2024-07-18 "
          f"(workers={args.workers}, dry_run={args.dry_run})")
    print(f"Input root: {args.input_root}")

    client = None if args.dry_run else _make_client()
    model = "openai/gpt-4o-mini-2024-07-18"

    t0 = time.time()
    overall = {
        "files_done": 0, "records_done": 0, "records_skipped": 0,
        "records_error": 0, "label_counts": {},
    }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_file, p, client, model, args.dry_run): p
            for p in files
        }
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                s: CellStats = fut.result()
            except Exception as e:
                print(f"  file {p.name} FAILED with {type(e).__name__}: {e}")
                continue
            overall["files_done"] += 1
            overall["records_done"] += s.n_done
            overall["records_skipped"] += s.n_skipped
            overall["records_error"] += s.n_error
            for k, v in (s.label_counts or {}).items():
                overall["label_counts"][k] = overall["label_counts"].get(k, 0) + v

            elapsed = time.time() - t0
            rate = overall["records_done"] / elapsed if elapsed > 0 else 0
            print(
                f"[{overall['files_done']:>2}/{len(files)}] {p.name}  "
                f"+{s.n_done}done +{s.n_skipped}skip +{s.n_error}err  "
                f"labels={s.label_counts}  rate={rate:.1f}/s"
            )

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"DONE in {elapsed:.0f}s")
    print(f"  files_done={overall['files_done']}/{len(files)}")
    print(f"  records_done={overall['records_done']}")
    print(f"  records_skipped={overall['records_skipped']}")
    print(f"  records_error={overall['records_error']}")
    print(f"  label_counts={overall['label_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
