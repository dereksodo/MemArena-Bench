"""Re-judge D6 records in G3a (reader-side norm-binding) cells under out/g3a_access_marker_*.

G3a cells were run with --d6-inject-access (oracle × 5 readers × 3 seeds, d4_permission only).
The evaluate stage already populated ``policy_category`` via openai/gpt-4o-mini-2024-07-18 but
did not add ``rationale_v2``. This script fills rationale_v2 and harmonises all D6 fields using
the same 5-label rubric as the main pipeline.

Idempotent: each record is skipped if it already has a label in
``D6_LABELS`` and a non-empty ``rationale_v2``.

Run:
    python3 scripts/rerun_d6_judge_g3a.py [--max-files N] [--dry-run] [--workers N]
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
class FileStats:
    path: Path
    n_d4: int = 0
    n_done: int = 0
    n_skipped: int = 0
    n_error: int = 0
    label_counts: dict = None

    def __post_init__(self):
        self.label_counts = {}


def process_file(
    json_path: Path,
    client: Any,
    model: str,
    dry_run: bool = False,
) -> FileStats:
    from eval.src.scoring import D6_LABELS, _armB_correctness

    stats = FileStats(path=json_path)
    backup_path = json_path.with_name(json_path.stem + "_legacy.json")
    try:
        data = json.loads(json_path.read_text())
    except Exception as exc:
        print(f"  SKIP unreadable {json_path}: {exc}")
        return stats
    details = data.get("details", []) if isinstance(data, dict) else []
    d4_records = [(i, d) for i, d in enumerate(details)
                  if str(d.get("question_id", "")).startswith("d4_perm")]
    stats.n_d4 = len(d4_records)
    if not d4_records:
        return stats

    if not backup_path.exists() and not dry_run:
        shutil.copyfile(json_path, backup_path)

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
        ok, _r = _armB_correctness(label, expected_mode)
        if not dry_run:
            rec["policy_category"] = label
            rec["leaked_fact_in_output"] = fact_in
            rec["rationale_v2"] = rationale
            rec["reason"] = reason
            rec["correct"] = ok
            rec["score"] = 1.0 if ok else 0.0

        stats.n_done += 1
        if label == "PARSE_ERROR":
            stats.n_error += 1
        stats.label_counts[label] = stats.label_counts.get(label, 0) + 1

    if not dry_run:
        json_path.write_text(json.dumps(data, indent=2))
    return stats


def discover_g3a_files() -> list[Path]:
    """Find every evaluation_results JSON under out/g3a_access_marker_*/."""
    root = REPO_ROOT / "out"
    out: list[Path] = []
    for gdir in sorted(root.glob("g3a_access_marker_*")):
        if not gdir.is_dir():
            continue
        for p in gdir.rglob("evaluation_results*.json"):
            if p.stem.endswith("_legacy"):
                continue
            out.append(p)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    files = discover_g3a_files()
    if args.max_files:
        files = files[: args.max_files]
    if not files:
        print("No G3a files found under out/g3a_access_marker_*/.")
        return 1

    needs = 0
    for p in files:
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        for d in data.get("details", []):
            qid = str(d.get("question_id", ""))
            if not qid.startswith("d4_perm"):
                continue
            cat = str(d.get("policy_category") or "").upper()
            if cat in {"DISCLOSE_CORRECT", "DISCLOSE_WRONG", "DONT_KNOW", "REFUSE", "OTHER", "PARSE_ERROR"} \
               and d.get("rationale_v2"):
                continue
            needs += 1
    print(f"Discovered {len(files)} G3a files; {needs} records need re-judging "
          f"(workers={args.workers}, dry_run={args.dry_run}) via openai/gpt-4o-mini-2024-07-18")

    client = None if args.dry_run else _make_client()
    model = "openai/gpt-4o-mini-2024-07-18"

    t0 = time.time()
    overall = {"files_done": 0, "records_done": 0, "records_skipped": 0,
               "records_error": 0, "label_counts": {}}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_file, p, client, model, args.dry_run): p for p in files}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                s: FileStats = fut.result()
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
            print(f"[{overall['files_done']:>3}/{len(files)}] {p.relative_to(REPO_ROOT)}  "
                  f"+{s.n_done}done +{s.n_skipped}skip +{s.n_error}err  "
                  f"labels={s.label_counts}  rate={rate:.1f}/s")

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
