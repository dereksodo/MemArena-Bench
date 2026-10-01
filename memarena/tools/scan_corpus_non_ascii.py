#!/usr/bin/env python3
"""Scan a corpus JSONL for non-ASCII script contamination (hallucinated
Chinese / Russian / Japanese / Korean / Arabic / etc. in a supposedly
English dataset).

The LLM-generated MemArena corpus is meant to be English. Any Han,
Cyrillic, Kana, Hangul, Arabic, Hebrew, Greek, Devanagari, or Thai
character in it is almost certainly a model hallucination that slipped
through — e.g. GPT emitting "他说" mid-sentence, a name rendered as
"Иван" instead of "Ivan".

This tool:
  1. Reads the JSONL line by line.
  2. Walks every string value in each JSON record.
  3. Flags every character whose Unicode block is one of the "script"
     ranges listed below, classifies it, and records:
       - line number (1-indexed)
       - JSON path (dot/index notation, e.g., `turns[3].text`)
       - the offending character and its Unicode codepoint
       - a short surrounding-text snippet (±40 chars)
  4. Prints a compact summary grouped by script, then writes two reports:
       - <outdir>/non_ascii_report.json   (machine-readable, every hit)
       - <outdir>/non_ascii_report.md     (human-readable, grouped)

Punctuation-class non-ASCII (smart quotes, em-dash, ellipsis, degree
sign, accented-Latin like é/ñ/ö, mathematical ≤/≥/α) is NOT flagged —
these are legitimate English typography and not hallucinations.

Usage:
  python scripts/scan_corpus_non_ascii.py <file.jsonl> [--out DIR]
                                           [--limit-per-line N]
                                           [--max-examples N]

Defaults:
  file:   MASim/runs/l_20260408_111046/spark_results/cache/corpus_sessions_spark8.jsonl
  out:    docs/reviews/corpus_non_ascii/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FILE = (ROOT / "MASim" / "runs" / "l_20260408_111046"
                / "spark_results" / "cache"
                / "corpus_sessions_spark8.jsonl")
DEFAULT_OUT = ROOT / "docs" / "reviews" / "corpus_non_ascii"


# Unicode script ranges we treat as "real hallucinations" if found in an
# English corpus. (codepoint_start, codepoint_end_inclusive, label)
SCRIPT_RANGES = [
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x04FF, "Cyrillic"),
    (0x0500, 0x052F, "Cyrillic-Supplement"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0750, 0x077F, "Arabic-Supplement"),
    (0x0780, 0x07BF, "Thaana"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul-Jamo"),
    (0x1200, 0x137F, "Ethiopic"),
    (0x13A0, 0x13FF, "Cherokee"),
    (0x3040, 0x309F, "Hiragana"),
    (0x30A0, 0x30FF, "Katakana"),
    (0x3100, 0x312F, "Bopomofo"),
    (0x3130, 0x318F, "Hangul-Compatibility-Jamo"),
    (0x3200, 0x32FF, "CJK-Enclosed"),
    (0x3400, 0x4DBF, "CJK-Ext-A"),
    (0x4E00, 0x9FFF, "CJK-Unified"),
    (0xA000, 0xA4CF, "Yi"),
    (0xAC00, 0xD7AF, "Hangul-Syllables"),
    (0xF900, 0xFAFF, "CJK-Compatibility"),
    (0xFE30, 0xFE4F, "CJK-Compatibility-Forms"),
    (0xFF00, 0xFFEF, "Halfwidth-Fullwidth"),
    (0x1B000, 0x1B0FF, "Kana-Supplement"),
    (0x20000, 0x2A6DF, "CJK-Ext-B"),
    (0x2A700, 0x2B73F, "CJK-Ext-C"),
    (0x2B740, 0x2B81F, "CJK-Ext-D"),
    (0x2B820, 0x2CEAF, "CJK-Ext-E"),
]


def classify(cp: int) -> str | None:
    for lo, hi, label in SCRIPT_RANGES:
        if lo <= cp <= hi:
            return label
    return None


def walk_strings(obj, path: str):
    """Yield (path, value) for every string leaf in a nested JSON structure."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_strings(v, f"{path}[{i}]")
    # ignore int / float / bool / None


def snippet(text: str, idx: int, width: int = 40) -> str:
    lo = max(0, idx - width)
    hi = min(len(text), idx + width + 1)
    s = text[lo:hi]
    # collapse internal whitespace for compactness
    s = re.sub(r"\s+", " ", s)
    return s


def scan_value(text: str, path: str, limit_per_string: int = 8):
    """Yield hits in one string value."""
    hits = 0
    for idx, ch in enumerate(text):
        cp = ord(ch)
        if cp < 128:
            continue
        label = classify(cp)
        if label is None:
            continue
        yield {
            "path": path,
            "char": ch,
            "codepoint": f"U+{cp:04X}",
            "name": unicodedata.name(ch, "?"),
            "script": label,
            "index": idx,
            "snippet": snippet(text, idx),
        }
        hits += 1
        if hits >= limit_per_string:
            return


def scan_line(line_no: int, line: str, limit_per_line: int):
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        return [{
            "line": line_no,
            "path": "<parse-error>",
            "char": "",
            "codepoint": "",
            "name": "",
            "script": "JSON-parse-error",
            "snippet": str(e)[:200],
        }]
    hits = []
    for path, s in walk_strings(obj, ""):
        for h in scan_value(s, path):
            h2 = dict(h)
            h2["line"] = line_no
            hits.append(h2)
            if len(hits) >= limit_per_line:
                return hits
    return hits


def try_session_id(line: str) -> str | None:
    try:
        obj = json.loads(line)
    except Exception:
        return None
    for k in ("session_id", "id", "uid", "key", "day_idx"):
        if isinstance(obj, dict) and k in obj:
            return f"{k}={obj[k]}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", nargs="?", default=str(DEFAULT_FILE),
                    help="JSONL file to scan")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="Output directory")
    ap.add_argument("--limit-per-line", type=int, default=30,
                    help="Stop scanning a single line after this many hits "
                         "(default: 30)")
    ap.add_argument("--max-examples", type=int, default=10,
                    help="Max example hits shown per script in the summary "
                         "(default: 10)")
    args = ap.parse_args()

    src = Path(args.file)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        return 2

    by_script: dict[str, list[dict]] = defaultdict(list)
    per_script_counter: Counter[str] = Counter()
    total_lines = 0
    lines_with_hits = 0
    total_hits = 0
    all_hits: list[dict] = []

    with src.open() as fh:
        for line_no, line in enumerate(fh, 1):
            total_lines += 1
            if not line.strip():
                continue
            hits = scan_line(line_no, line, args.limit_per_line)
            if hits:
                lines_with_hits += 1
                sid = try_session_id(line)
                for h in hits:
                    if sid:
                        h["session_id_hint"] = sid
                    by_script[h["script"]].append(h)
                    per_script_counter[h["script"]] += 1
                    total_hits += 1
                all_hits.extend(hits)

    # Write JSON report (full)
    json_report = outdir / "non_ascii_report.json"
    json_report.write_text(json.dumps({
        "source": str(src),
        "lines_scanned": total_lines,
        "lines_with_hits": lines_with_hits,
        "total_hits": total_hits,
        "hits_by_script": dict(per_script_counter),
        "hits": all_hits,
    }, indent=2, ensure_ascii=False))

    # Write Markdown report (grouped summary + examples)
    md: list[str] = [
        "# Corpus non-ASCII script scan",
        "",
        f"- source: `{src}`",
        f"- lines scanned: {total_lines}",
        f"- lines with hits: {lines_with_hits}",
        f"- total script-character hits: {total_hits}",
        "",
    ]
    if not per_script_counter:
        md.append("**Result: CLEAN — no non-ASCII script characters found.**")
        md.append("")
        md.append("(Accented Latin letters, smart quotes, em-dashes, ellipses, "
                  "the micro sign, and math symbols are ignored by design — "
                  "they are legitimate English typography.)")
    else:
        md.append("## Hits by script")
        md.append("")
        md.append("| Script | Count | Distinct lines |")
        md.append("|---|---:|---:|")
        for script, n in sorted(per_script_counter.items(), key=lambda x: -x[1]):
            distinct_lines = len({h["line"] for h in by_script[script]})
            md.append(f"| {script} | {n} | {distinct_lines} |")
        md.append("")
        md.append(f"## Examples (up to {args.max_examples} per script)")
        for script, hits in sorted(by_script.items(), key=lambda x: -len(x[1])):
            md.append("")
            md.append(f"### {script} — {len(hits)} hit(s)")
            md.append("")
            seen = set()
            shown = 0
            for h in hits:
                key = (h["line"], h["path"])
                if key in seen:
                    continue
                seen.add(key)
                shown += 1
                if shown > args.max_examples:
                    break
                sid = h.get("session_id_hint", "")
                sid_part = f" [{sid}]" if sid else ""
                md.append(
                    f"- line {h['line']}{sid_part} · `{h['path']}` · "
                    f"`{h['codepoint']}` ({h['char']}): "
                    f"…{h['snippet']}…"
                )

    md_report = outdir / "non_ascii_report.md"
    md_report.write_text("\n".join(md) + "\n")

    # Console summary
    print(f"Scanned:        {total_lines} lines")
    print(f"Lines w/ hits:  {lines_with_hits}")
    print(f"Total hits:     {total_hits}")
    if per_script_counter:
        print("By script:")
        for s, n in sorted(per_script_counter.items(), key=lambda x: -x[1]):
            print(f"  {s:<32s} {n}")
    else:
        print("CLEAN — no script-class non-ASCII characters found.")
    print(f"\nReports:\n  {json_report}\n  {md_report}")
    return 0 if total_hits == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
