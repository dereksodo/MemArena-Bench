#!/usr/bin/env python3
"""Small helpers for the grid runners (scripts/test_cell.sh, judge_cell.sh).

The shell scripts call this for everything that is awkward in bash 3.2:
JSON state files, log parsing, free ports and dataset subsets. Standard
library only, so it runs with any python3 before the venv exists.

    grid_helper.py progress-set FILE key=value ...
    grid_helper.py progress-get FILE KEY
    grid_helper.py judge-set FILE TAG key=value ... # progress.json judges.<TAG> (+ legacy flat keys for gpt4omini)
    grid_helper.py judge-get FILE TAG KEY           # judges.<TAG>.KEY, legacy fallback for gpt4omini
    grid_helper.py judge-info TAG FIELD             # FIELD: model | file_tag
    grid_helper.py stage FILE N NAME
    grid_helper.py items FILE DONE TOTAL UNIT       # sets items + ETA, prints a status line
    grid_helper.py parse-log LOG answer|build       # prints "DONE TOTAL" or nothing
    grid_helper.py count-items DATASET [LIMIT]
    grid_helper.py make-limit-dataset SRC DST N
    grid_helper.py free-port START [SPAN]
    grid_helper.py cache-check SUMMARY_JSON
    grid_helper.py answers-check ANSWER_JSON EXPECTED
    grid_helper.py judge-check EVAL_JSON ANSWER_JSON
    grid_helper.py judge-mark EVAL_JSON DIMS        # mark paper dims (D1..D6, comma list) as not judged
    grid_helper.py judge-pending EVAL_JSON          # prints how many records wait for the judge
    grid_helper.py redo-dims DATASET DIMS           # task files of paper dims DIMS: "<n items> <file> ..."
    grid_helper.py splice-answers ANSWER_JSON NEW_JSON  # replace ANSWER_JSON's rows by NEW_JSON's (backup kept)
    grid_helper.py write-args FILE key=value ...    # credential-like keys are redacted
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memarena.judges import LEGACY_JUDGE, LEGACY_PROGRESS_KEYS, judge_spec, judge_state  # noqa: E402

SECRET_SUFFIXES = ("api_key", "_token", "secret", "password", "_key")
D6_LABELS = ("DISCLOSE_CORRECT", "DISCLOSE_WRONG", "DONT_KNOW", "REFUSE", "OTHER")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> float | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _coerce(value: str):
    if value == "null":
        return None
    if value in ("true", "false"):
        return value == "true"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    return value


def _load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".progress.", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def _pairs(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = _coerce(v)
    return out


def _fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


@contextlib.contextmanager
def _locked(path: Path):
    """Serialise read-modify-write of a progress file: two judges (e.g. the
    DeepSeek and the gpt-4o-mini run) may update the same cell at once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError:  # not POSIX: no locking
        yield
        return
    with open(path.with_name(path.name + ".lock"), "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def cmd_progress_set(path: str, *items: str) -> int:
    p = Path(path)
    with _locked(p):
        data = _load(p)
        data.update(_pairs(list(items)))
        data["updated_at"] = _now_iso()
        _save(p, data)
    return 0


def cmd_judge_set(path: str, tag: str, *items: str) -> int:
    """Update ``judges.<tag>`` only, never another judge's state. For the legacy
    judge (gpt4omini) the flat ``judge_*`` keys are mirrored as before."""
    judge_spec(tag)
    p = Path(path)
    pairs = _pairs(list(items))
    with _locked(p):
        data = _load(p)
        judges = data.get("judges") if isinstance(data.get("judges"), dict) else {}
        state = dict(judges.get(tag) or {})
        state.update(pairs)
        state["updated_at"] = _now_iso()
        judges[tag] = state
        data["judges"] = judges
        if tag == LEGACY_JUDGE:
            for key, value in pairs.items():
                if key in LEGACY_PROGRESS_KEYS:
                    data[LEGACY_PROGRESS_KEYS[key]] = value
        data["updated_at"] = _now_iso()
        _save(p, data)
    return 0


def _print_value(value) -> None:
    if value is None:
        print("")
    elif isinstance(value, bool):
        print("true" if value else "false")
    else:
        print(value)


def cmd_judge_get(path: str, tag: str, key: str) -> int:
    _print_value(judge_state(_load(Path(path)), tag).get(key))
    return 0


def cmd_judge_info(tag: str, field: str) -> int:
    spec = judge_spec(tag)
    if field not in ("model", "file_tag"):
        raise SystemExit(f"unknown judge field {field}")
    print(getattr(spec, field))
    return 0


def cmd_progress_get(path: str, key: str) -> int:
    _print_value(_load(Path(path)).get(key))
    return 0


def cmd_stage(path: str, n: str, name: str) -> int:
    """Close the current stage in the history and open stage N."""
    p = Path(path)
    data = _load(p)
    now = _now_iso()
    history = data.setdefault("stages", [])
    if history and history[-1].get("ended_at") is None:
        history[-1]["ended_at"] = now
        started = _parse_iso(history[-1].get("started_at"))
        history[-1]["elapsed_s"] = int(time.time() - started) if started else None
    if name != "-":
        history.append({"stage": int(n), "name": name, "started_at": now, "ended_at": None})
        data.update({
            "stage": f"{n}/{data.get('stage_total', 4)}",
            "stage_index": int(n),
            "stage_name": name,
            "stage_started_at": now,
            "items_done": None,
            "items_total": None,
            "items_unit": None,
            "eta_seconds": None,
            "eta": None,
        })
    data["updated_at"] = now
    _save(p, data)
    return 0


def cmd_items(path: str, done: str, total: str, unit: str) -> int:
    p = Path(path)
    data = _load(p)
    done_i, total_i = int(done), int(total)
    started = _parse_iso(data.get("stage_started_at")) or time.time()
    elapsed = time.time() - started
    eta_s = None
    if 0 < done_i < total_i:
        eta_s = int(elapsed * (total_i - done_i) / done_i)
    elif total_i and done_i >= total_i:
        eta_s = 0
    data.update({
        "items_done": done_i,
        "items_total": total_i,
        "items_unit": unit,
        "eta_seconds": eta_s,
        "eta": (datetime.fromtimestamp(time.time() + eta_s, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if eta_s is not None else None),
        "updated_at": _now_iso(),
    })
    _save(p, data)
    pct = 100.0 * done_i / total_i if total_i else 0.0
    print(f"{unit} {done_i}/{total_i} ({pct:.1f}%), elapsed {_fmt_dur(elapsed)}, ETA {_fmt_dur(eta_s)}")
    return 0


# Progress bars in eval/src/pipeline.py: "\r[eval_answer] |####----| 12/345  3.5%"
_BAR = re.compile(r"\[(eval_search|eval_answer)\] \|[#-]*\| (\d+)/(\d+)")
# scripts/reproduce/build_memory_cache.py: "[day 3/15] day=2 ..."
_DAY = re.compile(r"\[day (\d+)/(\d+)\]")


def cmd_parse_log(path: str, kind: str) -> int:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            text = f.read().decode("utf-8", "replace")
    except OSError:
        return 0
    if kind == "answer":
        hits = _BAR.findall(text)
        if hits:
            stage, done, total = hits[-1]
            print(f"{done} {total} {stage}")
    elif kind == "build":
        hits = _DAY.findall(text)
        if hits:
            done, total = hits[-1]
            print(f"{done} {total} days")
    else:
        raise SystemExit(f"unknown log kind {kind}")
    return 0


def _task_files(dataset: Path) -> list[Path]:
    return sorted((dataset / "eval_instances").glob("*.jsonl"))


def _rows(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_count_items(dataset: str, limit: str = "0") -> int:
    lim = int(limit or 0)
    total = 0
    files = _task_files(Path(dataset))
    if not files:
        raise SystemExit(f"no eval_instances/*.jsonl under {dataset}")
    for f in files:
        n = len(_rows(f))
        total += min(n, lim) if lim > 0 else n
    print(total)
    return 0


def cmd_make_limit_dataset(src: str, dst: str, n: str) -> int:
    """DST = SRC with every task file cut to its first N items.

    Everything else (corpus, personas, ego map, ...) is symlinked, so the
    subset differs from the full dataset only in which items are asked.
    """
    src_p, dst_p, lim = Path(src).resolve(), Path(dst), int(n)
    if lim <= 0:
        raise SystemExit("limit must be positive")
    tmp = dst_p.with_name(dst_p.name + f".tmp{os.getpid()}")
    (tmp / "eval_instances").mkdir(parents=True, exist_ok=True)
    for entry in src_p.iterdir():
        if entry.name == "eval_instances":
            continue
        os.symlink(entry, tmp / entry.name)
    for f in _task_files(src_p):
        rows = _rows(f)[:lim]
        (tmp / "eval_instances" / f.name).write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    (tmp / "LIMIT.json").write_text(json.dumps({"source": str(src_p), "limit_per_task_file": lim,
                                                "created_at": _now_iso()}, indent=2) + "\n")
    os.replace(tmp, dst_p)
    return 0


def _port_free(port: int) -> bool:
    for host in ("0.0.0.0", "127.0.0.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, port))
        except OSError:
            return False
        finally:
            s.close()
    return True


def cmd_free_port(start: str, span: str = "10") -> int:
    base = int(start)
    for port in range(base, base + int(span)):
        if _port_free(port):
            print(port)
            return 0
    print(f"no free port in {base}..{base + int(span) - 1}", file=sys.stderr)
    return 1


def cmd_cache_check(path: str) -> int:
    """Exit 0 when build_memory_cache.py finished and lost nothing."""
    try:
        s = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"no readable summary ({exc})")
        return 1
    ingest = s.get("ingest") or {}
    problems = []
    if s.get("status") != "complete":
        problems.append(f"status={s.get('status')} error={s.get('error')}")
    for key in ("messages_failed", "flush_errors", "add_exceptions"):
        if int(ingest.get(key) or 0):
            problems.append(f"{key}={ingest.get(key)}")
    if ingest and ingest.get("messages_succeeded") != ingest.get("messages_submitted"):
        problems.append(f"messages {ingest.get('messages_succeeded')}/{ingest.get('messages_submitted')} succeeded")
    if int(ingest.get("ego_days_unreported") or 0):
        problems.append(f"ego_days_unreported={ingest.get('ego_days_unreported')}")
    if problems:
        print("; ".join(problems))
        return 1
    print(f"complete: {s.get('n_rows_written')} rows ({s.get('n_errored')} error rows), "
          f"{ingest.get('messages_succeeded')}/{ingest.get('messages_submitted')} messages ingested, "
          f"{s.get('elapsed_seconds')}s")
    return 0


def cmd_answers_check(path: str, expected: str) -> int:
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"unreadable answer file ({exc})")
        return 1
    n = len(rows) if isinstance(rows, list) else -1
    failed = sum(1 for r in rows if isinstance(r, dict) and r.get("prediction") is None) if n >= 0 else 0
    think = sum(1 for r in rows if isinstance(r, dict) and r.get("think_detected")) if n >= 0 else 0
    print(f"{n} answers ({failed} failed calls, {think} with <think>), expected {expected}")
    return 0 if n == int(expected) else 1


def cmd_judge_check(eval_path: str, answer_path: str) -> int:
    """Exit 0 when every answer is scored and every D6 record carries the relabel."""
    try:
        data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
        answers = json.loads(Path(answer_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"unreadable ({exc})")
        return 1
    details = data.get("details") or []
    d6 = [d for d in details if str(d.get("question_id", "")).startswith("d4_perm")]
    relabelled = [d for d in d6 if str(d.get("policy_category") or "").upper() in D6_LABELS and d.get("rationale_v2") is not None]
    parse_err = sum(1 for d in d6 if str(d.get("policy_category") or "").upper() == "PARSE_ERROR")
    acc = (data.get("summary") or {}).get("accuracy")
    pending = _pending(details)
    print(f"judged {len(details)}/{len(answers)}; D6 relabelled {len(relabelled)}/{len(d6)} "
          f"(PARSE_ERROR {parse_err}); still marked for the judge {pending}; "
          f"accuracy {acc if acc is None else round(float(acc), 4)}")
    ok = len(details) == len(answers) and len(relabelled) == len(d6) and parse_err == 0 and pending == 0
    return 0 if ok else 1


# paper dimension -> question-id prefixes (memarena/figures/main_grid.PAPER_DIMS; D6 = d4_permission)
PAPER_DIM_PREFIXES = {
    "D1": ("d5",), "D2": ("d6",), "D3": ("d7", "d8", "d10"),
    "D4": ("d1", "d2"), "D5": ("d3",), "D6": ("d4",),
}


def _paper_dims(spec: str) -> list[str]:
    dims = [d.strip().upper() for d in spec.replace(" ", ",").split(",") if d.strip()]
    unknown = [d for d in dims if d not in PAPER_DIM_PREFIXES]
    if unknown or not dims:
        raise SystemExit(f"judge-mark: unknown dimension(s) {unknown or spec!r} (valid: D1..D6)")
    return dims


def cmd_judge_mark(eval_path: str, spec: str) -> int:
    """Mark every record of the given paper dimensions as not judged yet.

    Every marked record gets ``needs_judge``: scripts/llmjudge.py --only-marked
    re-judges exactly those from the answer file (so re-answered questions are
    judged on their new answers) and leaves the rest alone. D6 records also lose
    their five-label verdict, which scripts/rerun_d6_judge.py then relabels.
    """
    path = Path(eval_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    prefixes = {p: d for d in _paper_dims(spec) for p in PAPER_DIM_PREFIXES[d]}
    counts: dict[str, int] = {}
    for rec in data.get("details") or []:
        dim = prefixes.get(str(rec.get("question_id", "")).split("_")[0])
        if dim is None:
            continue
        if dim == "D6":
            # the relabel skips records that still carry a five-label verdict
            rec["policy_category"] = None
            rec["rationale_v2"] = None
            rec["leaked_fact_in_output"] = None
        # every marked record is re-read from the answer file (it may have been re-answered)
        rec["needs_judge"] = True
        counts[dim] = counts.get(dim, 0) + 1
    _save(path, data)
    print("marked " + ", ".join(f"{d} {n}" for d, n in sorted(counts.items())) if counts else "marked nothing")
    return 0


def cmd_redo_dims(dataset: str, spec: str) -> int:
    """Print the item count and the task files (stems) of the paper dimensions in ``spec``."""
    files, n = [], 0
    for dim in _paper_dims(spec):
        for prefix in PAPER_DIM_PREFIXES[dim]:
            for f in sorted(Path(dataset, "eval_instances").glob(f"{prefix}_*.jsonl")):
                files.append(f.stem)
                with f.open(encoding="utf-8") as fh:
                    n += sum(1 for line in fh if line.strip())
    if not files:
        raise SystemExit(f"redo-dims: no task files for {spec} under {dataset}/eval_instances")
    print(n, *files)
    return 0


def cmd_splice_answers(answer_path: str, new_path: str) -> int:
    """Replace the rows of an answer file by the re-answered rows of ``new_path``.

    Every re-answered question must already be in the answer file, so the file keeps
    its questions and order; the previous file is kept next to it as a backup.
    """
    old_rows = json.loads(Path(answer_path).read_text(encoding="utf-8"))
    new_rows = {r["question_id"]: r for r in json.loads(Path(new_path).read_text(encoding="utf-8"))}
    known = {r["question_id"] for r in old_rows}
    unknown = sorted(set(new_rows) - known)
    if unknown:
        raise SystemExit(f"splice-answers: {len(unknown)} re-answered questions are not in {answer_path}, e.g. {unknown[:3]}")
    backup = Path(answer_path).with_name(Path(answer_path).stem + f".before_redo_{time.strftime('%Y%m%d-%H%M%S')}.json")
    backup.write_text(Path(answer_path).read_text(encoding="utf-8"), encoding="utf-8")
    rows = [new_rows.get(r["question_id"], r) for r in old_rows]
    _save(Path(answer_path), rows)
    print(f"replaced {len(new_rows)} of {len(rows)} answers (previous file: {backup.name})")
    return 0


def _pending(details: list) -> int:
    return sum(1 for d in details if d.get("needs_judge"))


def cmd_judge_pending(eval_path: str) -> int:
    try:
        data = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print(0)
        return 0
    print(_pending(data.get("details") or []))
    return 0


def cmd_write_args(path: str, *items: str) -> int:
    data = _pairs(list(items))
    for k in list(data):
        if data[k] and k.lower().endswith(SECRET_SUFFIXES):
            data[k] = "<redacted>"
    data["written_at"] = _now_iso()
    _save(Path(path), data)
    return 0


COMMANDS = {
    "progress-set": cmd_progress_set,
    "progress-get": cmd_progress_get,
    "judge-set": cmd_judge_set,
    "judge-get": cmd_judge_get,
    "judge-info": cmd_judge_info,
    "stage": cmd_stage,
    "items": cmd_items,
    "parse-log": cmd_parse_log,
    "count-items": cmd_count_items,
    "make-limit-dataset": cmd_make_limit_dataset,
    "free-port": cmd_free_port,
    "cache-check": cmd_cache_check,
    "answers-check": cmd_answers_check,
    "judge-check": cmd_judge_check,
    "judge-mark": cmd_judge_mark,
    "judge-pending": cmd_judge_pending,
    "redo-dims": cmd_redo_dims,
    "splice-answers": cmd_splice_answers,
    "write-args": cmd_write_args,
}


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    return int(COMMANDS[argv[1]](*argv[2:]) or 0)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
