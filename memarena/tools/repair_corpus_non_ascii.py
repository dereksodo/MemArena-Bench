#!/usr/bin/env python3
"""Repair CJK / Cyrillic contamination in the MemArena-L corpus via GPT.

Reads a JSONL corpus (default: corpus_sessions_spark8.jsonl), finds every
string-valued leaf containing Han (CJK) characters, full-width CJK
punctuation, or Cyrillic homoglyphs, and asks GPT-5.4 to rewrite that
single string as natural English — preserving meaning, tone, emoji,
formatting, and [IMAGE:...] tags — with no non-ASCII script characters.

Greek-language quoted phrases are left untouched: in the current corpus
they're deliberate in-character bilingual dialogue (speaker says a
Greek phrase, then supplies an English gloss), not hallucinations.

Writes the repaired corpus back to the same path atomically, after
copying the original to a timestamped sibling:
    <original>.pre_ascii_fix.<YYYYMMDD_HHMMSS>.backup

A per-field change log lands in:
    docs/reviews/corpus_non_ascii/repair_log_<YYYYMMDD_HHMMSS>.md

Usage:
  python scripts/repair_corpus_non_ascii.py            # full repair
  python scripts/repair_corpus_non_ascii.py --dry-run  # nothing written
  python scripts/repair_corpus_non_ascii.py --limit 5  # first 5 lines only
  python scripts/repair_corpus_non_ascii.py --file <path> --model gpt-5.4
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FILE = (ROOT / "MASim" / "runs" / "l_20260408_111046"
                / "spark_results" / "cache"
                / "corpus_sessions_spark8.jsonl")
LOG_DIR = ROOT / "docs" / "reviews" / "corpus_non_ascii"
REFCHECKER_ENV = Path.home() / "Documents" / "refchecker" / ".env"


# Same ranges as the scanner — the subset we WILL rewrite.
SCRIPT_RANGES_TO_REWRITE = [
    (0x0400, 0x052F, "Cyrillic"),        # Cyrillic + supplement
    (0x3400, 0x4DBF, "CJK-Ext-A"),
    (0x4E00, 0x9FFF, "CJK-Unified"),
    (0xF900, 0xFAFF, "CJK-Compatibility"),
    (0xFE30, 0xFE4F, "CJK-Compatibility-Forms"),
    (0xFF00, 0xFFEF, "Halfwidth-Fullwidth"),
    (0x20000, 0x2A6DF, "CJK-Ext-B"),
]


SYSTEM_PROMPT = """You are repairing strings from a corpus of English
personal-assistant dialog that has been accidentally contaminated with
brief code-switches into Chinese (Han characters), full-width CJK
punctuation, or Cyrillic homoglyphs. These non-English insertions are
model hallucinations, not part of any speaker's persona.

Task
----
Rewrite the single input string so that:
 - All Han / Chinese characters are replaced with natural, idiomatic
   English that expresses the same meaning, in the same register (the
   corpus is casual, character-driven, often slang-heavy).
 - All full-width CJK punctuation (， ； ： 。 ？ ！ etc.) is replaced
   with the matching ASCII punctuation.
 - Any Cyrillic homoglyph accidentally substituted for a Latin letter
   (e.g., Cyrillic е U+0435 used in "bribе them") is replaced with the
   correct Latin letter.

Preserve EVERYTHING else exactly as it is:
 - Emoji (e.g., 🌊 😩 🤯) — keep verbatim, do not remove or change.
 - Smart quotes, em-dashes, ellipses, and other English-compatible
   typography.
 - Markup / tags like `[IMAGE: ...]`, `[sound: ...]`, Markdown-style
   bold/italic, bracketed stage directions. Keep them as-is; only
   translate their contents if those contents contain Chinese.
 - Surrounding English prose — do not rephrase English that was already
   English. Only substitute the non-English parts.
 - Line breaks, leading/trailing whitespace, and overall length (±20%).

Output
------
Output ONLY the repaired string. No surrounding quotes, no prose, no
markdown fences, no explanation. If the input is already ASCII-clean
and needed no change (shouldn't happen, but just in case), return it
unchanged.
"""


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k.strip(), v)


def classify(cp: int) -> str | None:
    for lo, hi, label in SCRIPT_RANGES_TO_REWRITE:
        if lo <= cp <= hi:
            return label
    return None


def contaminated(s: str) -> bool:
    return any(classify(ord(ch)) is not None for ch in s)


def walk_strings(obj, path: str):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_strings(v, f"{path}[{i}]")


def set_by_path(obj, path: str, new_value) -> None:
    """Set `obj`'s value at the dot/bracket `path` to `new_value` in-place."""
    tokens: list = []
    # parse path like "turns[3].metadata.thought"
    i = 0
    buf = ""
    while i < len(path):
        c = path[i]
        if c == ".":
            if buf:
                tokens.append(buf); buf = ""
            i += 1
        elif c == "[":
            if buf:
                tokens.append(buf); buf = ""
            j = path.index("]", i)
            tokens.append(int(path[i + 1:j]))
            i = j + 1
        else:
            buf += c
            i += 1
    if buf:
        tokens.append(buf)
    cur = obj
    for t in tokens[:-1]:
        cur = cur[t]
    cur[tokens[-1]] = new_value


def call_repair(client, model: str, s: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": s},
        ],
    )
    out = resp.choices[0].message.content or ""
    # Strip accidental surrounding quotes / fences
    out = out.strip()
    out = re.sub(r"^```[A-Za-z]*\s*\n", "", out)
    out = re.sub(r"\n```\s*$", "", out)
    return out.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", default=str(DEFAULT_FILE),
                    help="Corpus JSONL to repair")
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run the repair but do not write back")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N contaminated lines")
    ap.add_argument("--only-lines", default="",
                    help="Comma-separated list of 1-indexed line numbers")
    ap.add_argument("--concurrency", type=int, default=16,
                    help="Parallel GPT requests (default: 16)")
    args = ap.parse_args()

    load_env_file(REFCHECKER_ENV)
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    try:
        from openai import OpenAI  # type: ignore
    except Exception as ex:
        print(f"ERROR: openai SDK not importable: {ex}", file=sys.stderr)
        return 2
    client = OpenAI()

    src = Path(args.file)
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        return 2
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    only = {int(x) for x in args.only_lines.split(",") if x.strip()} or None
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = src.with_suffix(src.suffix + f".pre_ascii_fix.{ts}.backup")
    repaired_path = src.with_suffix(src.suffix + f".repaired.{ts}")
    log_path = LOG_DIR / f"repair_log_{ts}.md"

    # Read all lines
    raw_lines = src.read_text().splitlines(keepends=False)

    # Pass 1: decide which lines need repair
    tasks: list[tuple[int, dict, list[tuple[str, str]]]] = []
    for line_no, line in enumerate(raw_lines, 1):
        if not line.strip():
            continue
        if only is not None and line_no not in only:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[line {line_no}] JSON parse error, skipping: {e}",
                  file=sys.stderr)
            continue
        field_edits: list[tuple[str, str]] = []
        for path, s in walk_strings(obj, ""):
            if contaminated(s):
                field_edits.append((path, s))
        if field_edits:
            tasks.append((line_no, obj, field_edits))
            if args.limit and len(tasks) >= args.limit:
                break

    print(f"Source:   {src}")
    print(f"Model:    {args.model}")
    print(f"Lines to repair: {len(tasks)}")
    total_fields = sum(len(t[2]) for t in tasks)
    print(f"Fields to repair: {total_fields}")
    print(f"Dry run:  {args.dry_run}")
    if not args.dry_run:
        print(f"Backup:   {backup}")
    print()

    log: list[str] = [
        f"# Corpus ASCII-repair log — {ts}",
        f"- source: `{src}`",
        f"- model: `{args.model}`",
        f"- dry_run: {args.dry_run}",
        f"- lines with contamination: {len(tasks)}",
        f"- fields repaired: {total_fields}",
        "",
    ]

    # Build flat job list so we can fan them out across threads.
    jobs: list[tuple[int, str, str]] = []
    for line_no, obj, edits in tasks:
        for path, original in edits:
            jobs.append((line_no, path, original))

    results: dict[tuple[int, str], str] = {}
    failures: list[tuple[int, str, str]] = []  # (line, path, reason)
    t0 = time.time()

    def worker(job):
        line_no, path, original = job
        try:
            repaired = call_repair(client, args.model, original)
        except Exception as ex:
            return (line_no, path, original, None, f"API {type(ex).__name__}: {ex}")
        if not repaired:
            return (line_no, path, original, None, "EMPTY reply")
        return (line_no, path, original, repaired, None)

    total = len(jobs)
    done = 0
    partial_count = 0
    print(f"Firing {total} GPT requests at concurrency={args.concurrency}")
    print()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, j) for j in jobs]
        for fut in as_completed(futures):
            line_no, path, original, repaired, err = fut.result()
            done += 1
            if err or repaired is None:
                failures.append((line_no, path, err or "UNKNOWN"))
                print(f"  [{done:04d}/{total}] line {line_no:>4d} · {path} · "
                      f"FAIL {err}", flush=True)
                continue
            still = contaminated(repaired)
            if still:
                partial_count += 1
            results[(line_no, path)] = repaired
            if done % 25 == 0 or done == total:
                print(f"  [{done:04d}/{total}] "
                      f"ok={len(results)} fail={len(failures)} "
                      f"partial={partial_count}", flush=True)

    # Stitch results into each line's JSON object.
    line_to_new_obj: dict[int, dict] = {}
    for line_no, obj, edits in tasks:
        new_obj = copy.deepcopy(obj)
        any_applied = False
        log.append(f"\n## line {line_no}\n")
        for path, original in edits:
            repaired = results.get((line_no, path))
            if repaired is None:
                log.append(f"### `{path}` — FAILED (no repair applied)\n")
                continue
            set_by_path(new_obj, path, repaired)
            any_applied = True
            tag = " (partial — residual script chars)" if contaminated(repaired) else ""
            log.append(
                f"### `{path}`{tag}\n"
                f"**before:**\n```\n{original}\n```\n\n"
                f"**after:**\n```\n{repaired}\n```\n"
            )
        if any_applied:
            line_to_new_obj[line_no] = new_obj

    elapsed = time.time() - t0
    failures_count = len(failures)
    log.append(f"\n---\nTotal time: {elapsed:.1f}s, failures: {failures_count}\n")
    log_path.write_text("\n".join(log))
    print(f"\nLog:    {log_path}")

    if args.dry_run:
        print("Dry-run: not writing repaired corpus.")
        return 0

    # Backup original
    shutil.copy2(src, backup)
    print(f"Backup: {backup}")

    # Emit repaired corpus
    with repaired_path.open("w") as fh:
        for line_no, raw in enumerate(raw_lines, 1):
            if line_no in line_to_new_obj:
                fh.write(json.dumps(line_to_new_obj[line_no],
                                    ensure_ascii=False) + "\n")
            else:
                fh.write(raw + "\n")
    print(f"Written: {repaired_path}")

    # Atomically replace the original
    shutil.move(str(repaired_path), src)
    print(f"Replaced: {src}")
    print(f"\nDone. Lines repaired: {len(line_to_new_obj)}; "
          f"field failures: {failures_count}; elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
